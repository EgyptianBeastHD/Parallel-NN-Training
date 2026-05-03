#include "allreduce.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace allreduce {

const char* algorithm_name(Algorithm a) {
    switch (a) {
        case Algorithm::TREE:             return "tree";
        case Algorithm::RING:             return "ring";
        case Algorithm::HALVING_DOUBLING: return "hd";
        case Algorithm::MPI_BUILTIN:      return "mpi";
    }
    return "?";
}

Algorithm parse_algorithm(const std::string& s) {
    if (s == "tree")                              return Algorithm::TREE;
    if (s == "ring")                              return Algorithm::RING;
    if (s == "hd" || s == "halving_doubling")     return Algorithm::HALVING_DOUBLING;
    if (s == "mpi" || s == "builtin")             return Algorithm::MPI_BUILTIN;
    throw std::invalid_argument("unknown all-reduce algorithm: " + s);
}

// ---------------------------------------------------------------------------
// Tree (binomial reduce + binomial broadcast)
// ---------------------------------------------------------------------------
//
// Reduce to rank 0 in log(P) steps. At step k (mask = 1 << k):
//     ranks with bit k clear:  receive from rank | mask, accumulate
//     ranks with bit k set:    send to rank & ~mask, then drop out
// The broadcast is the same pattern in reverse.
//
// Non-power-of-two P works without modification because the partner check
// is just `partner < size`. Bytes moved per rank: ~ 2 * log(P) * N * sizeof(double).

static void binomial_reduce_to_root(double* buf, std::size_t count, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    if (size == 1) return;

    std::vector<double> tmp(count);
    int  mask   = 1;
    bool active = true;
    while (mask < size && active) {
        if ((rank & mask) == 0) {
            int src = rank | mask;
            if (src < size) {
                MPI_Recv(tmp.data(), (int)count, MPI_DOUBLE, src, /*tag*/100,
                         comm, MPI_STATUS_IGNORE);
                for (std::size_t i = 0; i < count; ++i) buf[i] += tmp[i];
            }
        } else {
            int dst = rank & ~mask;
            MPI_Send(buf, (int)count, MPI_DOUBLE, dst, /*tag*/100, comm);
            active = false;
        }
        mask <<= 1;
    }
}

static void binomial_broadcast_from_root(double* buf, std::size_t count, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    if (size == 1) return;

    // Walk masks from 1 upward. After the step with mask = M, ranks [0, 2M)
    // hold the data: ranks [0, M) just sent to ranks [M, 2M).
    int mask = 1;
    while (mask < size) {
        if (rank < mask) {
            int dst = rank + mask;
            if (dst < size) {
                MPI_Send(buf, (int)count, MPI_DOUBLE, dst, /*tag*/101, comm);
            }
        } else if (rank < (mask << 1)) {
            int src = rank - mask;
            MPI_Recv(buf, (int)count, MPI_DOUBLE, src, /*tag*/101,
                     comm, MPI_STATUS_IGNORE);
        }
        mask <<= 1;
    }
}

void tree(double* buf, std::size_t count, MPI_Comm comm) {
    binomial_reduce_to_root(buf, count, comm);
    binomial_broadcast_from_root(buf, count, comm);
}

// ---------------------------------------------------------------------------
// Ring (reduce-scatter + all-gather, both around a unidirectional ring)
// ---------------------------------------------------------------------------
//
// Buffer is split into P chunks. The ring carries one chunk per step.
//
//   reduce-scatter:  step s = 0..P-2
//       send chunk[(rank - s)     mod P] -> next
//       recv chunk[(rank - s - 1) mod P] <- prev, accumulate into local
//
//   all-gather:      step s = 0..P-2
//       send chunk[(rank - s + 1) mod P] -> next
//       recv chunk[(rank - s)     mod P] <- prev (overwrite)
//
// At end of reduce-scatter rank r owns the fully-reduced chunk r.
// Bytes moved per rank: ~ 2 * (P-1)/P * N * sizeof(double) (bandwidth optimal).

void ring(double* buf, std::size_t count, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    if (size == 1) return;

    // Split into 'size' chunks of nearly equal length.
    std::vector<std::size_t> off(size + 1);
    for (int i = 0; i <= size; ++i) off[i] = (count * (std::size_t)i) / size;
    auto chunk_len = [&](int i) -> std::size_t { return off[i + 1] - off[i]; };

    int next = (rank + 1) % size;
    int prev = (rank - 1 + size) % size;

    std::size_t max_chunk = 0;
    for (int i = 0; i < size; ++i) max_chunk = std::max(max_chunk, chunk_len(i));
    std::vector<double> recv_buf(max_chunk);

    // Reduce-scatter
    for (int s = 0; s < size - 1; ++s) {
        int send_idx = (rank - s + size) % size;
        int recv_idx = (rank - s - 1 + size) % size;

        MPI_Sendrecv(buf + off[send_idx], (int)chunk_len(send_idx), MPI_DOUBLE, next, 200,
                     recv_buf.data(),     (int)chunk_len(recv_idx), MPI_DOUBLE, prev, 200,
                     comm, MPI_STATUS_IGNORE);

        std::size_t base = off[recv_idx];
        std::size_t n    = chunk_len(recv_idx);
        for (std::size_t i = 0; i < n; ++i) buf[base + i] += recv_buf[i];
    }

    // All-gather
    for (int s = 0; s < size - 1; ++s) {
        int send_idx = (rank - s + 1 + size) % size;
        int recv_idx = (rank - s     + size) % size;

        MPI_Sendrecv(buf + off[send_idx], (int)chunk_len(send_idx), MPI_DOUBLE, next, 201,
                     buf + off[recv_idx], (int)chunk_len(recv_idx), MPI_DOUBLE, prev, 201,
                     comm, MPI_STATUS_IGNORE);
    }
}

// ---------------------------------------------------------------------------
// Halving-doubling (Rabenseifner)
// ---------------------------------------------------------------------------
//
// Power-of-two P only. We pad the buffer up to a multiple of P with zeros
// so the recursive halves split cleanly.
//
//   recursive halving reduce-scatter (log P steps):
//     step k partner = rank XOR (1 << (logP - 1 - k))
//     exchange the half of the current [lo, hi) that the partner needs,
//     accumulate the half they sent us. After log P steps, rank r owns
//     chunk r at offset [r * N/P, (r+1) * N/P).
//
//   recursive doubling all-gather (log P steps):
//     step k partner = rank XOR (1 << k)
//     exchange the contiguous range we have for the contiguous range they have.
//     Ranges double in size each step.
//
// Bytes per rank: ~ 2 * (P-1)/P * N * sizeof(double), in 2 * log(P) latency steps.
// Best of both worlds for large messages on power-of-2 rank counts.

void halving_doubling(double* buf, std::size_t count, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    if (size == 1) return;

    if ((size & (size - 1)) != 0) {
        // Non-power-of-2: defer to MPI_Allreduce. Implementing the classic
        // Rabenseifner "fold extra ranks first" trick is straightforward but
        // not the focus of the comparison.
        MPI_Allreduce(MPI_IN_PLACE, buf, (int)count, MPI_DOUBLE, MPI_SUM, comm);
        return;
    }

    int log_size = 0;
    while ((1 << log_size) < size) ++log_size;

    // Pad up to a multiple of size with zeros.
    std::size_t padded     = ((count + size - 1) / size) * size;
    bool        needs_pad  = padded != count;
    std::vector<double> padded_buf;
    double* work = buf;
    if (needs_pad) {
        padded_buf.assign(padded, 0.0);
        std::copy(buf, buf + count, padded_buf.begin());
        work = padded_buf.data();
    }

    std::vector<double> recv_buf(padded);

    // Recursive halving reduce-scatter
    std::size_t lo = 0, hi = padded;
    for (int k = 0; k < log_size; ++k) {
        int partner = rank ^ (1 << (log_size - 1 - k));

        std::size_t mid = lo + (hi - lo) / 2;
        std::size_t send_lo, send_hi, recv_lo, recv_hi;
        if (rank < partner) {
            send_lo = mid; send_hi = hi;
            recv_lo = lo;  recv_hi = mid;
        } else {
            send_lo = lo;  send_hi = mid;
            recv_lo = mid; recv_hi = hi;
        }

        MPI_Sendrecv(work + send_lo, (int)(send_hi - send_lo), MPI_DOUBLE, partner, 300,
                     recv_buf.data(), (int)(recv_hi - recv_lo), MPI_DOUBLE, partner, 300,
                     comm, MPI_STATUS_IGNORE);

        std::size_t n = recv_hi - recv_lo;
        for (std::size_t i = 0; i < n; ++i) work[recv_lo + i] += recv_buf[i];

        lo = recv_lo;
        hi = recv_hi;
    }

    // Recursive doubling all-gather
    for (int k = 0; k < log_size; ++k) {
        int partner = rank ^ (1 << k);

        std::size_t span = hi - lo;
        std::size_t their_lo, their_hi;
        if (rank < partner) {
            their_lo = hi;        their_hi = hi + span;
        } else {
            their_lo = lo - span; their_hi = lo;
        }

        MPI_Sendrecv(work + lo,       (int)span, MPI_DOUBLE, partner, 301,
                     work + their_lo, (int)span, MPI_DOUBLE, partner, 301,
                     comm, MPI_STATUS_IGNORE);

        lo = std::min(lo, their_lo);
        hi = std::max(hi, their_hi);
    }

    if (needs_pad) {
        std::copy(padded_buf.begin(), padded_buf.begin() + count, buf);
    }
}

// ---------------------------------------------------------------------------
// Built-in baseline
// ---------------------------------------------------------------------------
void mpi_builtin(double* buf, std::size_t count, MPI_Comm comm) {
    MPI_Allreduce(MPI_IN_PLACE, buf, (int)count, MPI_DOUBLE, MPI_SUM, comm);
}

void run(Algorithm alg, double* buf, std::size_t count, MPI_Comm comm) {
    switch (alg) {
        case Algorithm::TREE:             tree(buf, count, comm);             return;
        case Algorithm::RING:             ring(buf, count, comm);             return;
        case Algorithm::HALVING_DOUBLING: halving_doubling(buf, count, comm); return;
        case Algorithm::MPI_BUILTIN:      mpi_builtin(buf, count, comm);      return;
    }
}

MPI_Comm dup_comm(MPI_Comm parent) {
    MPI_Comm out;
    MPI_Comm_dup(parent, &out);
    return out;
}

// ---------------------------------------------------------------------------
// Non-blocking pipelined helper
// ---------------------------------------------------------------------------
//
// Implementing custom non-blocking versions of tree/ring/HD requires manual
// state machines over MPI_Isend/Irecv. For the project we use MPI_Iallreduce,
// which lets the implementation choose its algorithm and lets us focus on
// the pipelining of *per-layer* gradients with backward-pass computation.

void istart(double* buf, std::size_t count, MPI_Comm comm, NbHandle& h) {
    MPI_Iallreduce(MPI_IN_PLACE, buf, (int)count, MPI_DOUBLE, MPI_SUM, comm, &h.req);
}

void iwait(NbHandle& h) {
    if (h.req != MPI_REQUEST_NULL) {
        MPI_Wait(&h.req, MPI_STATUS_IGNORE);
        h.req = MPI_REQUEST_NULL;
    }
}

void iwait_all(std::vector<NbHandle>& hs) {
    std::vector<MPI_Request> reqs;
    reqs.reserve(hs.size());
    for (auto& h : hs) {
        if (h.req != MPI_REQUEST_NULL) reqs.push_back(h.req);
    }
    if (!reqs.empty()) {
        MPI_Waitall((int)reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
    }
    for (auto& h : hs) h.req = MPI_REQUEST_NULL;
}

} // namespace allreduce
