#include "ann/ivf_flat.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace {
using Clock = std::chrono::steady_clock;
using Args = std::unordered_map<std::string, std::string>;

Args read_args(int argc, char** argv, int first) {
    Args args;
    for (int i = first; i < argc; i += 2) {
        const std::string name(argv[i]);
        if (name.rfind("--", 0) != 0 || i + 1 >= argc)
            throw std::runtime_error("Expected --option value pairs");
        if (!args.emplace(name.substr(2), argv[i + 1]).second)
            throw std::runtime_error("Duplicate option: " + name);
    }
    return args;
}

std::string required(const Args& args, const std::string& key) {
    const auto it = args.find(key);
    if (it == args.end()) throw std::runtime_error("Missing required --" + key);
    return it->second;
}

std::size_t natural(const Args& args, const std::string& key, std::size_t fallback, bool allow_zero = false) {
    const auto it = args.find(key);
    if (it == args.end()) return fallback;
    std::size_t parsed = 0;
    unsigned long long number = 0;
    try { number = std::stoull(it->second, &parsed); }
    catch (...) { throw std::runtime_error("Invalid integer for --" + key); }
    if (parsed != it->second.size() || it->second.empty() || it->second[0] == '-' ||
        (!allow_zero && number == 0) || number > std::numeric_limits<std::size_t>::max())
        throw std::runtime_error("Invalid integer for --" + key);
    return static_cast<std::size_t>(number);
}

void usage() {
    std::cout << "P24 sequential IVF-Flat ANN (squared L2)\n"
              << "  ann_cli build --input vectors.csv --output index.ivf --clusters 4 [--iterations 20] [--train-size 20000] [--seed 42]\n"
              << "  ann_cli search --index index.ivf --query 0.1,0.2 --topk 3 --nprobe 2\n"
              << "  ann_cli evaluate --index index.ivf --queries queries.csv --topk 3 --nprobe 2 [--limit 100]\n"
              << "CSV: one ID followed by numeric coordinates per row; no header.\n";
}

void print_neighbors(const ann::Dataset& data, const std::vector<ann::Neighbor>& results) {
    std::cout << "rank,id,squared_l2,l2\n";
    for (std::size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        std::cout << (i + 1) << ',' << data.ids[r.row_index] << ','
                  << std::setprecision(9) << r.squared_distance << ','
                  << std::sqrt(r.squared_distance) << '\n';
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2 || std::string(argv[1]) == "--help" || std::string(argv[1]) == "help") {
            usage();
            return argc < 2 ? 1 : 0;
        }
        const std::string command(argv[1]);
        const auto args = read_args(argc, argv, 2);
        if (command == "build") {
            auto data = ann::read_csv(required(args, "input"));
            ann::BuildOptions options;
            options.clusters = natural(args, "clusters", 4);
            options.iterations = natural(args, "iterations", 20);
            options.train_size = natural(args, "train-size", 20000, true);
            options.seed = natural(args, "seed", 42, true);
            const auto start = Clock::now();
            auto index = ann::IvfFlatIndex::build(std::move(data), options);
            const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
            index.save(required(args, "output"));
            std::cerr << "Built IVF-Flat index: vectors=" << index.dataset().size()
                      << " dimensions=" << index.dataset().dimensions
                      << " clusters=" << index.clusters()
                      << " train_seconds=" << seconds << '\n';
            return 0;
        }
        if (command == "search") {
            auto index = ann::IvfFlatIndex::load(required(args, "index"));
            auto query = ann::parse_vector(required(args, "query"));
            const auto k = natural(args, "topk", 10);
            const auto nprobe = natural(args, "nprobe", 1);
            const auto start = Clock::now();
            auto result = index.search(query, k, nprobe);
            const double microseconds = std::chrono::duration<double, std::micro>(Clock::now() - start).count();
            print_neighbors(index.dataset(), result);
            std::cerr << "search_us=" << microseconds << " results=" << result.size() << '\n';
            return 0;
        }
        if (command == "evaluate") {
            auto index = ann::IvfFlatIndex::load(required(args, "index"));
            auto queries = ann::read_csv(required(args, "queries"));
            if (queries.dimensions != index.dataset().dimensions)
                throw std::runtime_error("Query and index dimensions differ");
            const auto k = natural(args, "topk", 10);
            const auto nprobe = natural(args, "nprobe", 1);
            const auto limit = std::min(natural(args, "limit", queries.size()), queries.size());
            double recall = 0.0;
            double ann_us = 0.0;
            double exact_us = 0.0;
            for (std::size_t i = 0; i < limit; ++i) {
                std::vector<float> query(queries.row(i), queries.row(i) + queries.dimensions);
                auto start = Clock::now();
                const auto approximate = index.search(query, k, nprobe);
                ann_us += std::chrono::duration<double, std::micro>(Clock::now() - start).count();
                start = Clock::now();
                const auto exact = ann::exact_topk(index.dataset(), query, k);
                exact_us += std::chrono::duration<double, std::micro>(Clock::now() - start).count();
                recall += ann::recall_at_k(approximate, exact, k);
            }
            std::cout << std::fixed << std::setprecision(4)
                      << "queries=" << limit << '\n'
                      << "recall_at_" << k << '=' << recall / limit << '\n'
                      << "ann_mean_us=" << ann_us / limit << '\n'
                      << "exact_mean_us=" << exact_us / limit << '\n'
                      << "search_speedup_vs_exact=" << (ann_us > 0.0 ? exact_us / ann_us : 0.0) << '\n';
            return 0;
        }
        usage();
        throw std::runtime_error("Unknown command: " + command);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
