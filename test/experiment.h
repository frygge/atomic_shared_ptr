//
// Created by Joerg P. Schaefer on 01.09.2023.
//

#pragma once

#include <cstddef>
#include <vector>
#include <chrono>
#include <thread>
#include <atomic>
#include <latch>
#include <barrier>


namespace jps {

class experiment {
public:
    experiment( size_t n_workers,
                auto run_time = std::chrono::seconds ( 1 ),
                auto warmup_time = std::chrono::milliseconds ( 100 )) :
            n_workers_( n_workers ),
            sync_( n_workers + 1 ),
            run_time_( run_time ),
            warmup_time_( warmup_time ),
            continue_( true ),
            worker_scores_( n_workers )
    {}

    template<typename TestFunction>
    size_t run( const TestFunction& test_function ) {
        // start workers
        for( auto i = 0u; i < n_workers_; ++i ) {
            worker_scores_[i].hits.store( 0, std::memory_order_release );
            workers_.emplace_back( [&]( size_t worker_id ) {
                _get_worker_id() = worker_id;

                // synchronize with other workers
                sync_.arrive_and_wait();

                // go until we're supposed to stop
                while( continue_.test( std::memory_order_release )) {
                    test_function();
                    worker_scores_[worker_id].hits.fetch_add( 1, std::memory_order_seq_cst );
                }
            }, i );
        }

        return _run_and_finis();
    }
    template<class Derived>
    size_t run( void( Derived::*test_function )() ) {
        static_assert( std::is_base_of_v<experiment, Derived> );

        // start workers
        for( auto i = 0u; i < n_workers_; ++i ) {
            worker_scores_[i].hits.store( 0, std::memory_order_seq_cst );
            workers_.emplace_back( [&]( size_t worker_id ) {
                _get_worker_id() = worker_id;

                // synchronize with other workers
                sync_.arrive_and_wait();

                // go until we're supposed to stop
                while( continue_.test( std::memory_order_seq_cst )) {
                    ( static_cast<Derived*>( this )->*test_function )();
                    worker_scores_[worker_id].hits.fetch_add( 1, std::memory_order_seq_cst );
                }
            }, i );
        }

        return _run_and_finis();
    }
    template<class C>
    size_t run( C* c, void( C::*test_function )() ) {
        // start workers
        for( auto i = 0u; i < n_workers_; ++i ) {
            worker_scores_[i].hits.store( 0, std::memory_order_release );
            workers_.emplace_back( [&]( size_t worker_id ) {
                _get_worker_id() = worker_id;

                // synchronize with other workers
                sync_.arrive_and_wait();

                // go until we're supposed to stop
                while( continue_.test( std::memory_order_release )) {
                    c->*test_function();
                    worker_scores_[worker_id].hits.fetch_add( 1, std::memory_order_seq_cst );
                }
            }, i );
        }

        return _run_and_finis();
    }

protected:
    const size_t n_workers_;

    static size_t get_worker_id() {
        return _get_worker_id();
    }

private:
    static size_t& _get_worker_id() {
        static thread_local size_t worker_id;
        return worker_id;
    }
    size_t _run_and_finis() {
        // synchronize with workers
        std::this_thread::yield();
        sync_.arrive_and_wait();

        // let the threads start and warm up (e.g. converge in their caching behaviour)
        std::this_thread::sleep_for( warmup_time_ );
        size_t warmup_result = 0;
        for( auto i = 0u; i < n_workers_; ++i )
            warmup_result += worker_scores_[i].hits.load( std::memory_order_acquire );

        // let the workers do their job
        std::this_thread::sleep_for( run_time_ );

        // notify to finish the execution and gather the results
        continue_.clear( std::memory_order_release );
        size_t result = 0;
        for( auto i = 0u; i < n_workers_; ++i )
            result += worker_scores_[i].hits.load( std::memory_order_acquire );
        result -= warmup_result;

        // wait for workers to finish
        for( auto& w: workers_ )
            w.join();

        return result;
    }

    struct alignas( 128 ) worker_score {
        std::atomic<size_t> hits;
    };

    std::barrier<> sync_;
    std::vector<std::thread> workers_;
    std::chrono::duration<long double, std::nano> run_time_;
    std::chrono::duration<long double, std::nano> warmup_time_;

    std::atomic_flag continue_;
    std::vector<worker_score> worker_scores_;
};

}
