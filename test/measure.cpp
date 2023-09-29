//
// Created by Joerg P. Schaefer on 19.08.2023.
//

#define FOLLY_NO_CONFIG

#define MEASURE_STD
//#define MEASURE_VTYULB
//#define MEASURE_JSS
//#define MEASURE_FOLLY
#define MEASURE_JPS

#define MEASURE_STORE
#define MEASURE_LOAD
#define MEASURE_EXCHANGE
#define MEASURE_CAS_WEAK
#define MEASURE_CAS_STRONG
#define MEASURE_CAS_WEAK_LOOP
#define MEASURE_CAS_STRONG_LOOP

#include <chrono>
#include <memory>
#include <iostream>
#include <atomic>
#include <vector>
#include "shared_ptr.h"
#include "experiment.h"

// Anthony William's version
//#include "jss/atomic_shared_ptr.h"
// Facebooks version
//#include "folly/concurrency/AtomicSharedPtr.h"
// Vladisla Tyulbashev's version
//#include "atomic_shared_ptr.h"


using namespace std::chrono_literals;

bool measure_std = true;
bool measure_jss = true;
bool measure_folly = true;
bool measure_vtyulb = true;
bool measure_aios = true;

bool measure_store = true;
bool measure_load = true;
bool measure_exchange = true;
bool measure_cas_weak = true;
bool measure_cas_strong = true;
bool measure_cas_weak_loop = true;
bool measure_cas_strong_loop = true;

bool measure_with_contention = true;
bool measure_without_contention = true;

size_t min_workers = 1;
size_t max_workers = 48;
size_t min_vars = 1;
size_t max_vars = 64;

struct test {
    test( uint64_t u ) : u{ u }
    {}
    uint64_t u;
};

template<class ASPTR, bool contention = true>
class SptrExperiment : public jps::experiment
{
public:
    SptrExperiment( size_t n_workers, size_t n_vars, auto run_time = 1.0s ) :
            jps::experiment( n_workers, run_time, 0.1s ),
            atomic_sptrs_( contention? n_vars:n_workers )
    {}

protected:
    struct alignas( 128 ) atomic_sptr {
        ASPTR asp_;
    };
    std::vector<atomic_sptr> atomic_sptrs_;
};

template<class SPTR, class ASPTR, bool contention = true>
class e_store : public SptrExperiment<ASPTR, contention> {
public:
    e_store( size_t n_workers, size_t n_vars, auto run_time = 1.0s ) :
            SptrExperiment<ASPTR, contention>( n_workers, n_vars, run_time )
    {}
    size_t run() {
        return SptrExperiment<ASPTR, contention>::experiment::run( &e_store<SPTR, ASPTR, contention>::shoot );
    }
    void shoot() {
        static thread_local SPTR sptr{ new test{ this->get_worker_id() } };
        static thread_local size_t target = contention? 0 :this->get_worker_id();
        target = contention? ( target+1 ) % this->atomic_sptrs_.size() : this->get_worker_id();

        this->atomic_sptrs_[target].asp_.store( sptr, std::memory_order_release );
    }
};

template<class SPTR, class ASPTR, bool contention = true>
class e_load : public SptrExperiment<ASPTR, contention> {
public:
    e_load( size_t n_workers, size_t n_vars, auto run_time = 1.0s ) :
            SptrExperiment<ASPTR, contention>( n_workers, n_vars, run_time )
    {
        for( auto i = 0u; i < this->atomic_sptrs_.size(); ++i ) {
            SPTR p{ (
                            contention &&
                            n_vars > 3 &&
                            i == 0
                    )? nullptr : new test{ i }};
            this->atomic_sptrs_[i].asp_.store( p, std::memory_order_release );
        }
    }
    size_t run() {
        return SptrExperiment<ASPTR, contention>::experiment::run( &e_load<SPTR, ASPTR, contention>::shoot );
    }
    void shoot() {
        static thread_local size_t target = contention? 0 :this->get_worker_id();
        if( contention )
            target = ( target+1 ) % this->atomic_sptrs_.size();

        this->atomic_sptrs_[target].asp_.load( std::memory_order_acquire );
    }
};

template<class SPTR, class ASPTR, bool contention = true>
class e_exchange : public SptrExperiment<ASPTR, contention> {
public:
    e_exchange( size_t n_workers, size_t n_vars, auto run_time = 1.0s ) :
            SptrExperiment<ASPTR, contention>( n_workers, n_vars, run_time )
    {
        for( auto i = 0u; i < this->atomic_sptrs_.size(); ++i )
            this->atomic_sptrs_[i].asp_.store( SPTR{ new test{ i*2 }} );
    }
    size_t run() {
        return SptrExperiment<ASPTR, contention>::experiment::run( &e_exchange<SPTR, ASPTR, contention>::shoot );
    }
    void shoot() {
        static thread_local SPTR sptr{ new test{ this->get_worker_id()*2+1 } };
        static thread_local size_t target = contention? 0 :this->get_worker_id();
        target = contention? ( target+1 ) % this->atomic_sptrs_.size() : this->get_worker_id();

        sptr = this->atomic_sptrs_[target].asp_.exchange( std::move( sptr ), std::memory_order_release );
    }
};

template<class SPTR, class ASPTR, bool contention = true>
class e_cas_weak_loop : public SptrExperiment<ASPTR, contention> {
public:
    e_cas_weak_loop( size_t n_workers, size_t n_vars, auto run_time = 1.0s ) :
            SptrExperiment<ASPTR, contention>( n_workers, n_vars, run_time )
    {
        for( auto i = 0u; i < this->atomic_sptrs_.size(); ++i )
            this->atomic_sptrs_[i].asp_.store( SPTR{ new test{ i*2 }} );
    }
    size_t run() {
        return SptrExperiment<ASPTR, contention>::experiment::run( &e_cas_weak_loop<SPTR, ASPTR, contention>::shoot );
    }
    void shoot() {
        static thread_local SPTR sptr{ new test{ this->get_worker_id()*2+1 } };
        static thread_local size_t target = contention? 0 :this->get_worker_id();

        SPTR exp;
        while( !this->atomic_sptrs_[target].asp_.compare_exchange_weak(
                exp, sptr, std::memory_order_release, std::memory_order_acquire ))
            ;
        target = contention? ( target+ ( exp? exp->u:1ul ) ) % this->atomic_sptrs_.size() : this->get_worker_id();
        sptr = std::move( exp );
    }
};

template<class SPTR, class ASPTR, bool contention = true>
class e_cas_strong_loop : public SptrExperiment<ASPTR, contention> {
public:
    e_cas_strong_loop( size_t n_workers, size_t n_vars, auto run_time = 1.0s ) :
            SptrExperiment<ASPTR, contention>( n_workers, n_vars, run_time )
    {
        for( auto i = 0u; i < this->atomic_sptrs_.size(); ++i )
            this->atomic_sptrs_[i].asp_.store( SPTR{ new test{ i*2 }} );
    }
    size_t run() {
        return SptrExperiment<ASPTR, contention>::experiment::run( &e_cas_strong_loop<SPTR, ASPTR, contention>::shoot );
    }
    void shoot() {
        static thread_local SPTR sptr{ new test{ this->get_worker_id()*2+1 } };
        static thread_local size_t target = contention? 0 :this->get_worker_id();

        SPTR exp;
        while( !this->atomic_sptrs_[target].asp_.compare_exchange_strong(
                exp, sptr, std::memory_order_release, std::memory_order_acquire ))
            ;
        target = contention? ( target+ ( exp? exp->u:1ul ) ) % this->atomic_sptrs_.size() : this->get_worker_id();
        sptr = std::move( exp );
    }
};

template<class SPTR, class ASPTR, bool contention = true>
class e_cas_weak : public SptrExperiment<ASPTR, contention> {
public:
    e_cas_weak( size_t n_workers, size_t n_vars, auto run_time = 1.0s ) :
            SptrExperiment<ASPTR, contention>( n_workers, n_vars, run_time )
    {}
    size_t run() {
        return SptrExperiment<ASPTR, contention>::experiment::run( &e_cas_weak<SPTR, ASPTR, contention>::shoot );
    }
    void shoot() {
        static thread_local SPTR sptr{ new test{ this->get_worker_id()*2+1 } };
        static thread_local size_t target = 0;
        target = contention? ( target+1 ) % this->atomic_sptrs_.size() : this->get_worker_id();

        SPTR exp;
        this->atomic_sptrs_[target].asp_.compare_exchange_weak( exp, sptr, std::memory_order_acq_rel );
    }
};

template<class SPTR, class ASPTR, bool contention = true>
class e_cas_strong : public SptrExperiment<ASPTR, contention> {
    std::vector<SPTR> sptrs_;
public:
    e_cas_strong( size_t n_workers, size_t n_vars, auto run_time = 1.0s ) :
            SptrExperiment<ASPTR, contention>( n_workers, n_vars, run_time )
    {}
    size_t run() {
        return SptrExperiment<ASPTR, contention>::experiment::run( &e_cas_strong<SPTR, ASPTR, contention>::shoot );
    }
    void shoot() {
        static thread_local SPTR sptr{ new test{ this->get_worker_id()*2+1 } };
        static thread_local size_t target = contention? 0 :this->get_worker_id();
        target = contention? ( target+1 ) % this->atomic_sptrs_.size() : this->get_worker_id();

        SPTR exp;
        this->atomic_sptrs_[target].asp_.compare_exchange_strong( exp, sptr, std::memory_order_acq_rel );
    }
};

double measure( size_t v, size_t t, size_t n, void (*test)( size_t, size_t, size_t ) ) {
    auto t1 = std::chrono::high_resolution_clock::now();
    test( v, t, n );
    auto t2 = std::chrono::high_resolution_clock::now();

    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::seconds;

    duration<double, std::micro> mus_double = t2 - t1;
    return t * n / mus_double.count();
}

template<class T>
void test_lib( const std::string& lib, size_t repeat ) {
    std::cout << "=== library: " << lib << "\n"
              << "vars\tthreads\tthroughput(ops/us)\n";
    for( auto v = min_vars; v <= max_vars; v += 1 ) {
        for( auto t = min_workers; t <= max_workers; ++t ) {
            size_t n_ops = 0;
            for( auto r = 0u; r < repeat; ++r ) {
                T test( t, v, 2000ms );
                n_ops += test.run();
            }
            // ops/100ms = ops/repeat  ==>  ops/s = 10*ops/repeat  ==>  ops/us = 10*ops/repeat/1'000'000 = ops/repeat/100'000
            std::cout << v << "\t" << t << "\t" << double( n_ops ) / ( repeat * 2'000'000. ) << std::endl;
        }
    }
    std::cout << std::endl;
}

template<template<class, class, bool> class T>
void test_op( size_t repeat ) {
#ifdef MEASURE_JPS
    if( measure_aios ) {
        if( measure_with_contention ) {
            std::cout << "=== contention: true\n";
            std::cout << "=== lock_free: " << jps::atomic_shared_ptr<test>::is_always_lock_free << "\n";
            test_lib<T<jps::shared_ptr<test>, jps::atomic_shared_ptr<test>, true>>(
                    "jps", repeat );
        }
        if( measure_without_contention ) {
            std::cout << "=== contention: false\n";
            std::cout << "=== lock_free: " << jps::atomic_shared_ptr<test>::is_always_lock_free << "\n";
            test_lib<T<jps::shared_ptr<test>, jps::atomic_shared_ptr<test>, false>>(
                    "jps", repeat );
        }
    }
#endif

#ifdef MEASURE_FOLLY
    if( measure_folly ) {
        if( measure_with_contention ) {
            std::cout << "=== contention: true\n";
            //std::cout << "=== jps: " << folly::atomic_shared_ptr<test>::is_always_lock_free << "\n";
            test_lib<T<std::shared_ptr<test>, folly::atomic_shared_ptr<test>, true>>(
                    "folly", repeat );
        }
        if( measure_without_contention ) {
            std::cout << "=== contention: false\n";
            //std::cout << "=== jps: " << folly::atomic_shared_ptr<test>::is_always_lock_free << "\n";
            test_lib<T<std::shared_ptr<test>, folly::atomic_shared_ptr<test>, false>>(
                    "folly", repeat );
        }
    }
#endif

#ifdef MEASURE_JSS
    if( measure_jss ) {
        if( measure_with_contention ) {
            std::cout << "=== contention: true\n";
            //std::cout << "=== jps: " << jss::atomic_shared_ptr<test>::is_always_lock_free << "\n";
            test_lib<T<jss::shared_ptr<test>, jss::atomic_shared_ptr<test>, true>>(
                    "jss", repeat );
        }
        if( measure_without_contention ) {
            std::cout << "=== contention: false\n";
            //std::cout << "=== jps: " << jss::atomic_shared_ptr<test>::is_always_lock_free << "\n";
            test_lib<T<jss::shared_ptr<test>, jss::atomic_shared_ptr<test>, false>>(
                    "jss", repeat );
        }
    }
#endif

#ifdef MEASURE_STD
    if( measure_std ) {
        if( measure_with_contention ) {
            std::cout << "=== contention: true\n";
            std::cout << "=== lock_free: " << std::atomic<std::shared_ptr<test>>::is_always_lock_free << "\n";
            test_lib<T<std::shared_ptr<test>, std::atomic<std::shared_ptr<test>>, true>>(
                    "std", repeat );
        }
        if( measure_without_contention ) {
            std::cout << "=== contention: false\n";
            std::cout << "=== lock_free: " << std::atomic<std::shared_ptr<test>>::is_always_lock_free << "\n";
            test_lib<T<std::shared_ptr<test>, std::atomic<std::shared_ptr<test>>, false>>(
                    "std", repeat );
        }
    }
#endif

#ifdef MEASURE_VTYULB
    if( measure_vtyulb ) {
        if( measure_with_contention ) {
            std::cout << "=== contention: true\n";
            test_lib<T<LFStructs::SharedPtr<test>, LFStructs::AtomicSharedPtr<test>, true>>(
                    "vtyulb", repeat );
        }
        if( measure_without_contention ) {
            std::cout << "=== contention: false\n";
            test_lib<T<LFStructs::SharedPtr<test>, LFStructs::AtomicSharedPtr<test>, false>>(
                    "vtyulb", repeat );
        }
    }
#endif
}

void run_all()
{
    const size_t repeat = 1;

#ifdef MEASURE_STORE
    if( measure_store ) {
        std::cout << "=== operation: store\n";
        test_op<e_store>( repeat );
    }
#endif

#ifdef MEASURE_LOAD
    if( measure_load ) {
        std::cout << "=== operation: load\n";
        test_op<e_load>( repeat );
    }
#endif

#ifdef MEASURE_EXCHANGE
    if( measure_exchange ) {
        std::cout << "=== operation: exchange\n";
        test_op<e_exchange>( repeat );
    }
#endif

#ifdef MEASURE_CAS_WEAK
    if( measure_cas_weak ) {
        std::cout << "=== operation: cas_weak\n";
        test_op<e_cas_weak>( repeat );
    }
#endif

#ifdef MEASURE_CAS_STRONG
    if( measure_cas_strong) {
        std::cout << "=== operation: cas_strong\n";
        test_op<e_cas_strong>( repeat );
    }
#endif

#ifdef MEASURE_CAS_WEAK_LOOP
    if( measure_cas_weak_loop ) {
        std::cout << "=== operation: cas_weak_loop\n";
        test_op<e_cas_weak_loop>( repeat );
    }
#endif

#ifdef MEASURE_CAS_STRONG_LOOP
    if( measure_cas_strong_loop ) {
        std::cout << "=== operation: cas_strong_loop\n";
        test_op<e_cas_strong_loop>( repeat );
    }
#endif
}

int main( int argc, char* argv[] ) {
    for( auto i = 1; i < argc; ++i ) {
        const auto s = std::string( argv[i] );

        if( s == "-std" )
            measure_std = false;
        else if( s == "-jss" )
            measure_jss = false;
        else if( s == "-folly" )
            measure_folly = false;
        else if( s == "-vtyulb" )
            measure_vtyulb = false;
        else if( s == "-jps" )
            measure_aios = false;
        else if( s == "-default_lib" ) {
            measure_std = false;
            measure_vtyulb = false;
            measure_jss = false;
            measure_folly = false;
            measure_aios = false;
        }

        else if( s == "-store" )
            measure_store = false;
        else if( s == "-load" )
            measure_load = false;
        else if( s == "-exchange" )
            measure_exchange = false;
        else if( s == "-cas_weak" )
            measure_cas_weak = false;
        else if( s == "-cas_strong" )
            measure_cas_strong = false;
        else if( s == "-cas_weak_loop" )
            measure_cas_weak_loop = false;
        else if( s == "-cas_strong_loop" )
            measure_cas_strong_loop = false;
        else if( s == "-default_op" ) {
            measure_store = false;
            measure_load = false;
            measure_exchange = false;
            measure_cas_weak = false;
            measure_cas_strong = false;
            measure_cas_weak_loop = false;
            measure_cas_strong_loop = false;
        }

        else if( s == "+std" )
            measure_std = true;
        else if( s == "+jss" )
            measure_jss = true;
        else if( s == "+folly" )
            measure_folly = true;
        else if( s == "+vtyulb" )
            measure_vtyulb = true;
        else if( s == "+jps" )
            measure_aios = true;

        else if( s == "+store" )
            measure_store = true;
        else if( s == "+load" )
            measure_load = true;
        else if( s == "+exchange" )
            measure_exchange = true;
        else if( s == "+cas_weak" )
            measure_cas_weak = true;
        else if( s == "+cas_strong" )
            measure_cas_strong = true;
        else if( s == "+cas_weak_loop" )
            measure_cas_weak_loop = true;
        else if( s == "+cas_strong_loop" )
            measure_cas_strong_loop = true;

        else if( s == "-contention" )
            measure_with_contention = false;
        else if( s == "+contention" )
            measure_with_contention = true;
        else if( s == "-no_contention" )
            measure_without_contention = false;
        else if( s == "+no_contention" )
            measure_without_contention = true;

        else if( s == "-workers")
            min_workers = std::atoi( argv[++i] );
        else if( s == "-vars")
            min_vars = std::atoi( argv[++i] );
        else if( s == "+workers")
            max_workers = std::atoi( argv[++i] );
        else if( s == "+vars")
            max_vars = std::atoi( argv[++i] );

        else {
            std::cerr << "Unknown parameter: " << s << "\n";
            exit( -1 );
        }
    }

    run_all();
}
