/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

/*
 * c8_surrogate.h: the interface between CORSIKA 8 and a generative subshower model.
 *
 * A model is a shared library that exports one function, c8s_get_plugin, returning a
 * table of the functions below. CORSIKA loads it with dlopen at run time, so the model,
 * its inference code (ONNX Runtime, LibTorch, a GPU client, ...) and its weights live and
 * are built outside CORSIKA. This header is the only thing the two share. It is plain C,
 * so the library can be built with another compiler or C++ standard library than CORSIKA.
 *
 * Ownership: CORSIKA owns the particles it passes in; the model owns the hits it passes
 * out, which are valid until the sink returns. Nothing crosses the boundary otherwise.
 */
#ifndef C8_SURROGATE_H
#define C8_SURROGATE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Bumped whenever a struct or function below changes. CORSIKA refuses other versions. */
#define C8S_ABI_VERSION 1u

/*
 * One particle CORSIKA took out of the cascade, for the model to replace.
 *
 * Positions and momenta are in the model's frame, which the bundle's manifest defines
 * (corsika: frame), with the shower core where the run put it. Units: m, GeV/c, GeV, s.
 */
typedef struct {
  uint64_t seed; /* all of the model's randomness for this particle comes from it */
  uint64_t id;   /* per-shower counter, as in triggers.parquet */
  int32_t pdg;
  int32_t reserved; /* 0 */
  double x, y, z;
  double px, py, pz;
  double kinetic_energy; /* follows from p and pdg; passed for convenience */
  double time;           /* when CORSIKA made the particle */
  double weight;         /* thinning weight; CORSIKA multiplies the hits' weights by it */
} c8s_particle;

/*
 * One particle the model says enters one station of the array.
 *
 * Fields a model doesn't predict are NAN. The station response is applied afterwards,
 * to these together with the stations' Monte Carlo particles, so it isn't part of this.
 */
typedef struct {
  uint64_t particle; /* index of the c8s_particle in the batch passed to generate */
  int32_t station;   /* index into the bundle's station list */
  int32_t pdg;
  double kinetic_energy;
  double time;       /* absolute, like c8s_particle.time */
  double nx, ny, nz; /* direction of travel, in the model's frame */
  double weight;     /* multiplies the particle's weight: 1 unless the model resamples */
} c8s_hit;

/* Receives hits. May be called any number of times during one generate call. */
typedef void (*c8s_hit_sink)(void* context, c8s_hit const* hits, size_t n);

typedef struct c8s_model c8s_model; /* opaque, owned by the library */

typedef struct {
  uint32_t abi_version; /* C8S_ABI_VERSION as the library was built */
  char const* name;

  /* Load the bundle in bundle_dir. options: "key=value" pairs separated by commas or
     newlines, as given on CORSIKA's command line (threads, device, ...); may be empty.
     Returns NULL on failure, with a message in error. */
  c8s_model* (*create)(char const* bundle_dir, char const* options, char* error,
                       size_t error_size);

  void (*destroy)(c8s_model* model);

  /* YAML describing the loaded model (versions, settings), for CORSIKA's config.yaml.
     Owned by the model, valid until destroy. */
  char const* (*describe)(c8s_model const* model);

  /* Generate the hits of n particles, in any order, through sink. Each particle's hits
     must depend only on that particle (and its seed), not on the rest of the batch.
     Called by one thread at a time. Returns 0 on success; otherwise a message in error.
   */
  int (*generate)(c8s_model* model, c8s_particle const* particles, size_t n,
                  c8s_hit_sink sink, void* context, char* error, size_t error_size);
} c8s_plugin;

/* The function the library exports, under this name. */
typedef c8s_plugin const* (*c8s_get_plugin_fn)(void);
#define C8S_GET_PLUGIN_SYMBOL "c8s_get_plugin"

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* C8_SURROGATE_H */
