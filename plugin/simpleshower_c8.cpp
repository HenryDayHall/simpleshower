// libsimpleshower_c8: a SimpleShower bundle (see simpleshower/bundle.py) as a CORSIKA 8
// surrogate model, run with ONNX Runtime. It implements c8_surrogate.h, so CORSIKA
// loads it at run time:
//
//   c8_air_shower_surrogate ... --surrogate-bundle BUNDLE \
//       --surrogate-model libsimpleshower_c8.so --surrogate-options threads=4
//
// Options (comma separated): threads=N (ONNX Runtime's intra-op threads; default 1),
// max_batch=N (rows per stage-1 call; default 2048), max_tokens=N (deposit tokens per
// stage-2 call; default 2048), device=cpu|cuda.
//
// It does what simpleshower/sampling.py does, step for step, so the two give the same
// hits for the same particles and seeds: that is what test/test_bundle.py checks.

#include "c8_surrogate.h"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

  // ------------------------------------------------ per-particle noise (sampling.py)

  constexpr std::uint64_t golden = 0x9E3779B97F4A7C15ull;

  inline std::uint64_t mix(std::uint64_t z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }

  /// u_k of a seed: 24 bits, so exact in float32
  inline float uniform(std::uint64_t const seed, std::uint64_t const k) {
    return static_cast<float>(static_cast<double>(mix(seed + (k + 1) * golden) >> 40) *
                              0x1p-24);
  }

  // ------------------------------------------------------------------- options

  struct Options {
    int threads = 1;
    std::size_t maxBatch = 2048;
    std::size_t maxTokens = 2048;
    std::string device = "cpu";
  };

  Options parse_options(char const* text) {
    Options options;
    std::string s = text ? text : "";
    std::replace(s.begin(), s.end(), ',', '\n');
    std::istringstream lines(s);
    for (std::string line; std::getline(lines, line);) {
      line.erase(std::remove_if(line.begin(), line.end(), ::isspace), line.end());
      if (line.empty()) continue;
      auto const eq = line.find('=');
      if (eq == std::string::npos) throw std::runtime_error("option without '=': " + line);
      auto const key = line.substr(0, eq), value = line.substr(eq + 1);
      if (key == "threads") {
        options.threads = std::stoi(value);
      } else if (key == "max_batch") {
        options.maxBatch = std::stoul(value);
      } else if (key == "max_tokens") {
        options.maxTokens = std::stoul(value);
      } else if (key == "device") {
        options.device = value;
      } else {
        throw std::runtime_error("unknown option " + key);
      }
    }
    if (options.maxBatch == 0 || options.maxTokens == 0) {
      throw std::runtime_error("max_batch and max_tokens must be positive");
    }
    return options;
  }

  std::string metadata(Ort::Session const& session, char const* key) {
    Ort::AllocatorWithDefaultOptions allocator;
    auto const value =
        session.GetModelMetadata().LookupCustomMetadataMapAllocated(key, allocator);
    return value ? std::string(value.get()) : std::string();
  }

  int metadata_int(Ort::Session const& session, char const* key) {
    auto const text = metadata(session, key);
    if (text.empty()) throw std::runtime_error(std::string("graph has no ") + key);
    return std::stoi(text);
  }

} // namespace

// ----------------------------------------------------------------------- the model

struct c8s_model {
  explicit c8s_model(std::string const& bundle, Options const& o)
      : options(o)
      , env(ORT_LOGGING_LEVEL_WARNING, "simpleshower") {
    Ort::SessionOptions so;
    so.SetIntraOpNumThreads(options.threads);
    so.SetInterOpNumThreads(1);
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    if (options.device == "cuda") {
      so.AppendExecutionProvider_CUDA(OrtCUDAProviderOptions{});
    } else if (options.device != "cpu") {
      throw std::runtime_error("unknown device " + options.device);
    }
    multiplicity = std::make_unique<Ort::Session>(
        env, (bundle + "/multiplicity.onnx").c_str(), so);
    deposits = std::make_unique<Ort::Session>(env, (bundle + "/deposits.onnx").c_str(), so);

    for (auto const* session : {multiplicity.get(), deposits.get()}) {
      if (metadata_int(*session, "bundle_format") != 1) {
        throw std::runtime_error("bundle format isn't 1; re-export it, or update the plugin");
      }
    }
    noiseMultiplicity = metadata_int(*multiplicity, "noise_multiplicity");
    perDeposit = metadata_int(*deposits, "noise_per_deposit");
    std::istringstream codes(metadata(*multiplicity, "pdgs"));
    for (std::string code; std::getline(codes, code, ',');) {
      pdgs.push_back(std::stoi(code));
      index[pdgs.back()] = static_cast<std::int64_t>(pdgs.size()); // 0 is "other"
    }

    std::ostringstream d;
    d << "name: simpleshower\nbundle: " << bundle << "\nbundle_format: 1"
      << "\nonnxruntime: " << Ort::GetVersionString() << "\ndevice: " << options.device
      << "\nthreads: " << options.threads << "\nmax_batch: " << options.maxBatch
      << "\nmax_tokens: " << options.maxTokens << "\nincident_pdgs: [";
    for (std::size_t i = 0; i < pdgs.size(); ++i) d << (i ? ", " : "") << pdgs[i];
    d << "]\noutputs: [station, pdg, kinetic_energy]\n";
    description = d.str();
  }

  void generate(c8s_particle const* particles, std::size_t n, c8s_hit_sink sink,
                void* context);

  Options options;
  Ort::Env env;
  std::unique_ptr<Ort::Session> multiplicity, deposits;
  std::vector<std::int32_t> pdgs; // vocabulary index i + 1 is pdgs[i]
  std::unordered_map<std::int32_t, std::int64_t> index;
  int noiseMultiplicity = 2, perDeposit = 4;
  std::string description;
  Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
};

void c8s_model::generate(c8s_particle const* particles, std::size_t const n,
                         c8s_hit_sink sink, void* context) {
  // inputs of every particle, as sampling.py makes them
  std::vector<std::int64_t> pdgIndex(n);
  std::vector<float> kinematics(6 * n), noise(noiseMultiplicity * n);
  for (std::size_t i = 0; i < n; ++i) {
    auto const& p = particles[i];
    auto const found = index.find(p.pdg);
    pdgIndex[i] = found == index.end() ? 0 : found->second;
    double const k[6] = {p.x, p.y, p.z, p.px, p.py, p.pz};
    for (int c = 0; c < 6; ++c) kinematics[6 * i + c] = static_cast<float>(k[c]);
    for (int c = 0; c < noiseMultiplicity; ++c) {
      noise[noiseMultiplicity * i + c] = uniform(p.seed, c);
    }
  }

  // stage 1: how many deposits each subshower has
  std::vector<std::int64_t> multiplicityOf(n);
  char const* in1[] = {"pdg_index", "kinematics", "noise"};
  char const* out1[] = {"multiplicity", "p_none"};
  for (std::size_t start = 0; start < n; start += options.maxBatch) {
    auto const b = static_cast<std::int64_t>(std::min(options.maxBatch, n - start));
    std::int64_t const shape1[] = {b}, shape6[] = {b, 6},
                       shapeNoise[] = {b, noiseMultiplicity};
    std::vector<Ort::Value> inputs;
    inputs.push_back(Ort::Value::CreateTensor<std::int64_t>(
        memory, pdgIndex.data() + start, b, shape1, 1));
    inputs.push_back(Ort::Value::CreateTensor<float>(memory, kinematics.data() + 6 * start,
                                                     6 * b, shape6, 2));
    inputs.push_back(Ort::Value::CreateTensor<float>(
        memory, noise.data() + noiseMultiplicity * start, noiseMultiplicity * b,
        shapeNoise, 2));
    auto outputs =
        multiplicity->Run(Ort::RunOptions{nullptr}, in1, inputs.data(), 3, out1, 2);
    auto const* m = outputs[0].GetTensorData<std::int64_t>();
    std::copy(m, m + b, multiplicityOf.begin() + start);
  }

  // stage 2, for subshowers with deposits, grouped by multiplicity (sampling.buckets)
  std::vector<std::size_t> rows;
  for (std::size_t i = 0; i < n; ++i) {
    if (multiplicityOf[i] > 0) rows.push_back(i);
  }
  std::stable_sort(rows.begin(), rows.end(), [&](std::size_t a, std::size_t b) {
    return multiplicityOf[a] < multiplicityOf[b];
  });

  char const* in2[] = {"pdg_index", "kinematics", "multiplicity", "noise"};
  char const* out2[] = {"detector", "deposit_pdg_index", "log10_energy"};
  std::vector<c8s_hit> hits;
  auto run_group = [&](std::vector<std::size_t> const& group, std::int64_t const tokens) {
    auto const b = static_cast<std::int64_t>(group.size());
    std::vector<std::int64_t> gPdg(b), gMultiplicity(b);
    std::vector<float> gKinematics(6 * b), gNoise(b * tokens * perDeposit);
    for (std::int64_t r = 0; r < b; ++r) {
      std::size_t const i = group[r];
      gPdg[r] = pdgIndex[i];
      gMultiplicity[r] = multiplicityOf[i];
      std::copy_n(kinematics.begin() + 6 * i, 6, gKinematics.begin() + 6 * r);
      for (std::int64_t j = 0; j < tokens; ++j) {
        for (int c = 0; c < perDeposit; ++c) {
          gNoise[(r * tokens + j) * perDeposit + c] =
              uniform(particles[i].seed, noiseMultiplicity + perDeposit * j + c);
        }
      }
    }
    std::int64_t const shape1[] = {b}, shape6[] = {b, 6},
                       shapeNoise[] = {b, tokens, perDeposit};
    std::vector<Ort::Value> inputs;
    inputs.push_back(Ort::Value::CreateTensor<std::int64_t>(memory, gPdg.data(), b, shape1, 1));
    inputs.push_back(
        Ort::Value::CreateTensor<float>(memory, gKinematics.data(), 6 * b, shape6, 2));
    inputs.push_back(
        Ort::Value::CreateTensor<std::int64_t>(memory, gMultiplicity.data(), b, shape1, 1));
    inputs.push_back(Ort::Value::CreateTensor<float>(memory, gNoise.data(), gNoise.size(),
                                                     shapeNoise, 3));
    auto outputs = deposits->Run(Ort::RunOptions{nullptr}, in2, inputs.data(), 4, out2, 3);
    auto const* detector = outputs[0].GetTensorData<std::int64_t>();
    auto const* depositPdg = outputs[1].GetTensorData<std::int64_t>();
    auto const* logEnergy = outputs[2].GetTensorData<float>();
    hits.clear();
    for (std::int64_t r = 0; r < b; ++r) {
      for (std::int64_t j = 0; j < gMultiplicity[r]; ++j) { // the rest is padding
        auto const t = r * tokens + j;
        auto const v = depositPdg[t];
        hits.push_back(c8s_hit{group[r], static_cast<std::int32_t>(detector[t]),
                               v > 0 ? pdgs[v - 1] : 0,
                               std::pow(10.0, static_cast<double>(logEnergy[t])), NAN,
                               NAN, NAN, NAN, 1.0});
      }
    }
    sink(context, hits.data(), hits.size());
  };

  std::vector<std::size_t> group;
  std::int64_t largest = 0;
  for (std::size_t const row : rows) {
    auto const m = multiplicityOf[row];
    if (!group.empty() &&
        static_cast<std::size_t>(std::max(largest, m)) * (group.size() + 1) >
            options.maxTokens) {
      run_group(group, largest);
      group.clear();
      largest = 0;
    }
    group.push_back(row);
    largest = std::max(largest, m);
  }
  if (!group.empty()) run_group(group, largest);
}

// --------------------------------------------------------------- c8_surrogate.h

namespace {

  void write_error(char* error, std::size_t const size, char const* what) {
    if (error && size) std::snprintf(error, size, "%s", what);
  }

  c8s_model* create(char const* bundle, char const* options, char* error,
                    std::size_t const size) {
    try {
      return new c8s_model(bundle, parse_options(options));
    } catch (std::exception const& e) {
      write_error(error, size, e.what());
      return nullptr;
    }
  }

  void destroy(c8s_model* model) { delete model; }

  char const* describe(c8s_model const* model) { return model->description.c_str(); }

  int generate(c8s_model* model, c8s_particle const* particles, std::size_t const n,
               c8s_hit_sink sink, void* context, char* error, std::size_t const size) {
    try {
      model->generate(particles, n, sink, context);
      return 0;
    } catch (std::exception const& e) {
      write_error(error, size, e.what());
      return 1;
    }
  }

  c8s_plugin const plugin{C8S_ABI_VERSION, "simpleshower", create, destroy, describe,
                          generate};

} // namespace

extern "C" __attribute__((visibility("default"))) c8s_plugin const* c8s_get_plugin() {
  return &plugin;
}
