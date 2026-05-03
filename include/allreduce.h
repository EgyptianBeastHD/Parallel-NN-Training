#pragma once
// Hand-rolled MPI all-reduce algorithms for the CS5220 project.
// Drop-in replacement for MPI_Allreduce(MPI_IN_PLACE, ..., MPI_SUM).
//
//   tree              binomial tree reduce to rank 0 + binomial broadcast back
//                     latency:   2 * log(P) * alpha
//                     bandwidth: 2 * log(P) * N * beta
//                     best for:  small messages
//
//   ring              P-1 reduce-scatter steps + P-1 all-gather steps
//                     latency:   2 * (P-1) * alpha
//                     bandwidth: 2 * (P-1)/P * N * beta      (bandwidth optimal)
//                     best for:  large messages
//
//   halving_doubling  recursive-halving reduce-scatter +
//                     recursive-doubling all-gather (Rabenseifner)
//                     latency:   2 * log(P) * alpha
//                     bandwidth: 2 * (P-1)/P * N * beta      (bandwidth optimal)
//                     best for:  power-of-2 P, large messages
//                     falls back to MPI_Allreduce when P is not a power of two
//
//   mpi_builtin       reference baseline using MPI_Allreduce
//
// All functions perform an in-place sum reduction over MPI_DOUBLE.

#include <mpi.h>
#include <cstddef>
#include <string>
#include <vector>

namespace allreduce {

enum class Algorithm {
    TREE = 0,
    RING = 1,
    HALVING_DOUBLING = 2,
    MPI_BUILTIN = 3,
};

const char* algorithm_name(Algorithm a);
Algorithm   parse_algorithm(const std::string& s);

// ---- blocking variants (in place, sum reduction) -------------------------
void tree(double* buf, std::size_t count, MPI_Comm comm);
void ring(double* buf, std::size_t count, MPI_Comm comm);
void halving_doubling(double* buf, std::size_t count, MPI_Comm comm);
void mpi_builtin(double* buf, std::size_t count, MPI_Comm comm);

// dispatch by enum
void run(Algorithm alg, double* buf, std::size_t count, MPI_Comm comm);

// Returns a private communicator (dup of `parent`) that the caller should
// pass to the all-reduce primitives instead of MPI_COMM_WORLD. Isolating
// our point-to-point traffic in a separate communicator prevents
// implementation-defined ordering effects when the surrounding code also
// makes MPI calls (e.g. an MPI_Allreduce for loss reporting between
// iterations). The caller owns the returned communicator and must
// MPI_Comm_free it before MPI_Finalize.
MPI_Comm dup_comm(MPI_Comm parent);

// ---- non-blocking pipelined variant --------------------------------------
// Used to overlap per-layer gradient communication with the rest of the
// backward pass. We launch one MPI_Iallreduce per layer as the gradients
// for that layer become available, then wait on the full set just before
// the optimizer step.
struct NbHandle {
    MPI_Request req = MPI_REQUEST_NULL;
};

void istart(double* buf, std::size_t count, MPI_Comm comm, NbHandle& h);
void iwait(NbHandle& h);
void iwait_all(std::vector<NbHandle>& hs);

} // namespace allreduce
