/*
 * Copyright 2023 Joerg Peter Schaefer
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <type_traits>
#include <thread>
#include <memory>
#include <cassert>
#include <cstdint>
#include <atomic>

#define CACHE_COHERENCY_LINE_SIZE 64


namespace jps {

/*
 * Paired Counters
 */

struct paired_counter {
    constexpr paired_counter() noexcept : single_{ 0, 0 }
    {}
    constexpr paired_counter( int32_t c1, uint32_t c2 ) noexcept : single_{ c2, c1 }
    {}

    constexpr int32_t get_cnt1() const
    {
        return single_.cnt1;
    }
    constexpr uint32_t get_cnt2() const
    {
        return single_.cnt2;
    }
    constexpr void set_counter1( int32_t c )
    {
        single_.cnt1 = c;
    }
    constexpr void set_counter2( uint32_t c )
    {
        single_.cnt2 = c;
    }

    constexpr paired_counter& operator+=( const paired_counter& r ) noexcept
    {
        single_.cnt1 += r.single_.cnt1;
        single_.cnt2 += r.single_.cnt2;
        return *this;
    }
    constexpr paired_counter& operator-=( const paired_counter& r ) noexcept
    {
        single_.cnt1 -= r.single_.cnt1;
        single_.cnt2 -= r.single_.cnt2;
        return *this;
    }

    constexpr paired_counter operator+( const paired_counter& r ) const noexcept
    {
        return paired_counter{ get_cnt1()+r.get_cnt1(), get_cnt2()+r.get_cnt2() };
    }
    constexpr paired_counter operator+( paired_counter&& r ) const noexcept
    {
        return r += *this;
    }
    constexpr paired_counter operator-( const paired_counter& r ) const noexcept
    {
        return paired_counter{ get_cnt1()-r.get_cnt1(), get_cnt2()-r.get_cnt2() };
    }
    constexpr paired_counter operator-( paired_counter&& r ) const noexcept
    {
        return r -= *this;
    }

    constexpr bool operator==( const paired_counter& r ) const noexcept
    {
        return word() == r.word();
    }
    constexpr bool operator!=( const paired_counter& r ) const noexcept
    {
        return word() != r.word();
    }
    constexpr bool operator>( const paired_counter& r ) const noexcept
    {
        return get_cnt1() > r.get_cnt1()
               && get_cnt2() > r.get_cnt2();
    }
    constexpr bool operator>=( const paired_counter& r ) const noexcept
    {
        return get_cnt1() >= r.get_cnt1()
               && get_cnt2() >= r.get_cnt2();
    }
    constexpr bool operator<( const paired_counter& r ) const noexcept
    {
        return get_cnt1() < r.get_cnt1()
               && get_cnt2() < r.get_cnt2();
    }
    constexpr bool operator<=( const paired_counter& r ) const noexcept
    {
        return get_cnt1() <= r.get_cnt1()
               && get_cnt2() <= r.get_cnt2();
    }

    constexpr uint64_t word() const
    {
        return word_;
    }

private:
    friend struct atomic_paired_counter;
    constexpr paired_counter( uint64_t word ) noexcept : word_{ word }
    {}

    struct single {
        uint32_t cnt2;
        int32_t cnt1;
    };
    union {
        single single_;
        uint64_t word_;
    };
};

struct atomic_paired_counter {
    static constexpr auto is_always_lock_free = std::atomic<uint64_t>::is_always_lock_free;

    constexpr atomic_paired_counter() noexcept : counter_{ 0 }
    {}
    constexpr atomic_paired_counter( int32_t c1, uint32_t c2 ) : counter_{ paired_counter{ c1, c2 }.word() }
    {}
    constexpr atomic_paired_counter( const paired_counter& ctr2 ) : counter_{ ctr2.word() }
    {}
    atomic_paired_counter( const atomic_paired_counter& ) = delete;
    template<typename... ARGS>
    constexpr atomic_paired_counter( ARGS... args ) : counter_{ args... }
    {}

    bool is_lock_free() const noexcept
    {
        return counter_.is_lock_free();
    }

    void store( paired_counter desired, std::memory_order order = std::memory_order_seq_cst ) noexcept
    {
        counter_.store( desired.word_, order );
    }
    paired_counter load( std::memory_order order = std::memory_order_seq_cst ) const noexcept
    {
        return { counter_.load( order ) };
    }
    operator paired_counter() const noexcept
    {
        return load();
    }

    paired_counter exchange( paired_counter desired, std::memory_order order = std::memory_order_seq_cst ) noexcept
    {
        return { counter_.exchange( desired.word_, order ) };
    }

    bool compare_exchange_weak( paired_counter& expected, paired_counter desired,
                                std::memory_order success, std::memory_order failure ) noexcept
    {
        return counter_.compare_exchange_weak( expected.word_, desired.word_, success, failure );
    }
    bool compare_exchange_weak( paired_counter& expected, paired_counter desired,
                                std::memory_order order = std::memory_order_seq_cst ) noexcept
    {
        return counter_.compare_exchange_weak( expected.word_, desired.word_, order );
    }
    bool compare_exchange_weak_c1( int32_t& expected, int32_t desired,
                                   std::memory_order success, std::memory_order failure ) noexcept
    {
        auto cur_pair = paired_counter{ counter_.load( failure ) };
        if( cur_pair.get_cnt1() != expected ) {
            expected = cur_pair.get_cnt1();
            return false;
        }

        for(;;) {
            if( compare_exchange_weak( cur_pair, desired, success, failure ))
                return true;
            if( cur_pair.get_cnt1() != expected ) {
                expected = cur_pair.get_cnt1();
                return false;
            }
        }
    }
    bool compare_exchange_weak_c1( paired_counter& expected, int32_t desired,
                                   std::memory_order success, std::memory_order failure ) noexcept
    {
        return compare_exchange_weak( expected, { desired, expected.get_cnt2() }, success, failure );
    }
    bool compare_exchange_weak_c2( uint32_t& expected, uint32_t desired,
                                   std::memory_order success, std::memory_order failure ) noexcept
    {
        auto cur_pair = paired_counter{ counter_.load( failure ) };
        if( cur_pair.get_cnt2() != expected ) {
            expected = cur_pair.get_cnt2();
            return false;
        }

        for(;;) {
            if( compare_exchange_weak( cur_pair, desired, success, failure ))
                return true;
            if( cur_pair.get_cnt2() != expected ) {
                expected = cur_pair.get_cnt2();
                return false;
            }
        }
    }
    bool compare_exchange_weak_c2( paired_counter& expected, uint32_t desired,
                                   std::memory_order success, std::memory_order failure ) noexcept
    {
        return compare_exchange_weak( expected, { expected.get_cnt1(), desired }, success, failure );
    }

    bool compare_exchange_strong( paired_counter& expected, paired_counter desired,
                                  std::memory_order success, std::memory_order failure ) noexcept
    {
        return counter_.compare_exchange_strong( expected.word_, desired.word_, success, failure );
    }
    bool compare_exchange_strong( paired_counter& expected, paired_counter desired,
                                  std::memory_order order = std::memory_order_seq_cst ) noexcept
    {
        return counter_.compare_exchange_strong( expected.word_, desired.word_, order );
    }
    bool compare_exchange_strong_c1( int32_t& expected, int32_t desired,
                                     std::memory_order success, std::memory_order failure ) noexcept
    {
        auto cur_pair = paired_counter{ counter_.load( failure ) };
        if( cur_pair.get_cnt1() != expected ) {
            expected = cur_pair.get_cnt1();
            return false;
        }

        // the other counter was wrong, so try again
        return compare_exchange_strong( cur_pair, desired, success, failure );
    }
    bool compare_exchange_strong_c1( paired_counter& expected, int32_t desired,
                                     std::memory_order success, std::memory_order failure ) noexcept
    {
        for(;;) {
            auto cur_pair = expected;
            auto new_pair = paired_counter{ desired, expected.get_cnt2() };
            if( compare_exchange_strong( cur_pair, new_pair, success, failure ))
                return true;
            if( cur_pair.get_cnt1() != expected.get_cnt1() ) {
                expected = cur_pair;
                return false;
            }
        }
    }
    bool compare_exchange_strong_c2( uint32_t& expected, uint32_t desired,
                                     std::memory_order success, std::memory_order failure ) noexcept
    {
        auto cur_pair = paired_counter{ counter_.load( failure ) };
        if( cur_pair.get_cnt2() != expected ) {
            expected = cur_pair.get_cnt2();
            return false;
        }

        // the other counter was wrong, so try again
        return compare_exchange_strong( cur_pair, desired, success, failure );
    }
    bool compare_exchange_strong_c2( paired_counter& expected, uint32_t desired,
                                     std::memory_order success, std::memory_order failure ) noexcept
    {
        for(;;) {
            auto cur_pair = expected;
            auto new_pair = paired_counter{ expected.get_cnt1(), desired };
            if( compare_exchange_strong( cur_pair, new_pair, success, failure ))
                return true;
            if( cur_pair.get_cnt2() != expected.get_cnt2() ) {
                expected = cur_pair;
                return false;
            }
        }
    }

    void wait( paired_counter old, std::memory_order order = std::memory_order_seq_cst ) noexcept
    {
        return counter_.wait( old.word_, order );
    }
    void notify_one() noexcept
    {
        return counter_.notify_one();
    }
    void notify_all() noexcept
    {
        return counter_.notify_all();
    }

    paired_counter fetch_add( paired_counter arg, std::memory_order order = std::memory_order_seq_cst ) noexcept
    {
        return counter_.fetch_add( arg.word_, order );
    }
    paired_counter fetch_sub( paired_counter arg, std::memory_order order = std::memory_order_seq_cst ) noexcept
    {
        return counter_.fetch_sub( arg.word_, order );
    }
    paired_counter fetch_and( paired_counter arg, std::memory_order order = std::memory_order_seq_cst ) noexcept
    {
        return counter_.fetch_and( arg.word_, order );
    }
    paired_counter fetch_or( paired_counter arg, std::memory_order order = std::memory_order_seq_cst ) noexcept
    {
        return counter_.fetch_or( arg.word_, order );
    }
    paired_counter fetch_xor( paired_counter arg, std::memory_order order = std::memory_order_seq_cst ) noexcept
    {
        return counter_.fetch_xor( arg.word_, order );
    }

    /*
     * Atomically transfers an amount from c1 to c2.
     *
     * The new paired counter will be `{ c1-arg, c2+arg }`.
     */
    paired_counter fetch_transfer( int32_t arg, std::memory_order order = std::memory_order_seq_cst ) noexcept
    {
        if( arg >= 0 )
            return fetch_add( paired_counter{ -arg, uint32_t( arg ) }, order );

            // deal with overflow in c2
        else
            return fetch_sub( paired_counter{ arg, uint32_t( -arg ) }, order );
    }

private:
    union {
        paired_counter _pc_;
        std::atomic<uint64_t> counter_;
    };
    static_assert( std::atomic<uint64_t>::is_always_lock_free );
};

/*
 * Counted Pointers
 */

template<class T>
struct counted_ptr {
    uint64_t word_;

    constexpr static uint64_t ptr_mask = ( 1ul << 48 )-1ul;
    constexpr static uint64_t ctr_mask = ~0ul << 48;

    constexpr counted_ptr() noexcept : word_{ 0 }
    {}
    constexpr counted_ptr( uint64_t word ) noexcept : word_{ word }
    {}
    constexpr counted_ptr( int16_t counter, T* ptr ) noexcept : word_{ make_word( counter, ptr ) }
    {}
    constexpr counted_ptr( T* ptr ) noexcept : word_{ make_word( ptr ) }
    {}
    constexpr counted_ptr( int16_t c ) noexcept : word_{ make_word( c ) }
    {}

    static constexpr uint64_t make_word( int16_t counter ) noexcept {
        return static_cast<uint64_t>( counter ) << 48;
    }
    static constexpr uint64_t make_word( T* ptr ) noexcept {
        assert( ( reinterpret_cast<uint64_t>( ptr )&ctr_mask ) == 0 );
        return reinterpret_cast<uint64_t>( ptr );
    }
    static constexpr uint64_t make_word( int16_t counter, T* ptr ) noexcept {
        return make_word( counter ) | make_word( ptr );
    }

    constexpr int16_t& counter()
    {
        return static_cast<int16_t*>( static_cast<void*>( &word_ ))[3];
    }
    constexpr int16_t counter() const
    {
        return static_cast<const int16_t*>( static_cast<const void*>( &word_ ))[3];
    }
    constexpr int16_t get_ctr() const
    {
        return counter();
    }
    constexpr void set_ctr( int16_t c )
    {
        counter() = c;
    }

    constexpr T *get_ptr() const
    {
        return reinterpret_cast<T*>( word_ & ( ( 1ul << 48 )-1ul ) );
    }
    constexpr void set_ptr( T* p )
    {
        word_ = make_word( static_cast<const counted_ptr<T>*>( this )->get_ctr(), p );
    }

    constexpr T* operator->() const noexcept {
        return get_ptr();
    }
    constexpr T& operator*() const noexcept {
        return *get_ptr();
    }
};
static_assert( sizeof( counted_ptr<uint64_t> ) == sizeof( uint64_t ));


template<class T>
struct atomic_counted_ptr {
    using cptr_type = counted_ptr<T>;

    static constexpr auto is_always_lock_free = std::atomic<uint64_t>::is_always_lock_free;

    constexpr atomic_counted_ptr() noexcept : word_{ 0 }
    {}
    constexpr atomic_counted_ptr( const cptr_type& cptr ) : word_{ cptr.word_ }
    {}
    constexpr atomic_counted_ptr( int16_t counter, T* ptr ) noexcept : word_{ counted_ptr<T>::make_word( counter, ptr )}
    {}
    atomic_counted_ptr( const atomic_counted_ptr& ) = delete;

    T* get_ptr( std::memory_order order = std::memory_order_relaxed ) const noexcept
    {
        return counted_ptr<T>{ word_.load( order ) }.get_ptr();
    }
    int16_t get_ctr( std::memory_order order = std::memory_order_relaxed ) const noexcept
    {
        return counted_ptr<T>{ word_.load( order ) }.get_ctr();
    }

    cptr_type operator=( cptr_type desired ) noexcept
    {
        store( desired.word_, std::memory_order_relaxed );
        return desired;
    }
    atomic_counted_ptr& operator=( const atomic_counted_ptr& ) = delete;

    bool is_lock_free() const noexcept
    {
        return word_.is_lock_free();
    }

    constexpr void store( cptr_type desired, std::memory_order order = std::memory_order_seq_cst ) noexcept
    {
        word_.store( desired.word_, order );
    }
    constexpr cptr_type load( std::memory_order order = std::memory_order_seq_cst ) const noexcept
    {
        return { word_.load( order ) };
    }
    constexpr operator cptr_type() const noexcept
    {
        return load();
    }

    cptr_type exchange( cptr_type desired, std::memory_order order = std::memory_order_seq_cst ) noexcept
    {
        return { word_.exchange( desired.word_, order ) };
    }

    bool compare_exchange_weak( cptr_type& expected, cptr_type desired,
                                std::memory_order success, std::memory_order failure ) noexcept
    {
        return word_.compare_exchange_weak( expected.word_, desired.word_, success, failure );
    }
    bool compare_exchange_weak( cptr_type& expected, cptr_type desired,
                                std::memory_order order = std::memory_order_seq_cst ) noexcept
    {
        return word_.compare_exchange_weak( expected.word_, desired.word_, order );
    }
    bool compare_exchange_strong( cptr_type& expected, cptr_type desired,
                                  std::memory_order success, std::memory_order failure ) noexcept
    {
        return word_.compare_exchange_strong( expected.word_, desired.word_, success, failure );
    }
    bool compare_exchange_strong( cptr_type& expected, cptr_type desired,
                                  std::memory_order order = std::memory_order_seq_cst ) noexcept
    {
        return word_.compare_exchange_strong( expected.word_, desired.word_, order );
    }

    void wait( cptr_type old, std::memory_order order = std::memory_order_seq_cst ) noexcept
    {
        return word_.wait( reinterpret_cast<cptr_type&>( old ), order );
    }
    void notify_one() noexcept
    {
        return word_.notify_one();
    }
    void notify_all() noexcept
    {
        return word_.notify_all();
    }

    cptr_type fetch_add( int16_t arg, std::memory_order order = std::memory_order_seq_cst ) noexcept
    {
        return word_.fetch_add( counted_ptr<T>::make_word( arg ), order );
    }
    cptr_type fetch_sub( int16_t arg, std::memory_order order = std::memory_order_seq_cst ) noexcept
    {
        return word_.fetch_sub( counted_ptr<T>::make_word( arg ), order );
    }
    cptr_type fetch_and( int16_t arg, std::memory_order order = std::memory_order_seq_cst ) noexcept
    {
        return word_.fetch_and( counted_ptr<T>::make_word( arg, reinterpret_cast<T*>( counted_ptr<T>::ptr_mask )), order );
    }
    cptr_type fetch_or( int16_t arg, std::memory_order order = std::memory_order_seq_cst ) noexcept
    {
        return word_.fetch_or( counted_ptr<T>::make_word( arg ), order );
    }
    cptr_type fetch_xor( int16_t arg, std::memory_order order = std::memory_order_seq_cst ) noexcept
    {
        return word_.fetch_xor( counted_ptr<T>::make_word( arg ), order );
    }

    int16_t operator++() noexcept
    {
        return fetch_add( int16_t( 1 )).get_ctr() + 1;
    }
    int16_t operator++( int ) noexcept
    {
        return fetch_add( int16_t( 1 )).get_ctr();
    }
    int16_t operator--() noexcept
    {
        return fetch_sub( int16_t( 1 )).get_ctr() - 1;
    }
    int16_t operator--( int ) noexcept
    {
        return fetch_sub( int16_t( 1 )).get_ctr();
    }

    int16_t operator+=( int16_t arg ) noexcept
    {
        return fetch_add( arg ).get_ctr() + arg;
    }
    int16_t operator-=( int16_t arg ) noexcept
    {
        return fetch_sub( arg ).get_ctr() - arg;
    }
    int16_t operator&=( int16_t arg ) noexcept
    {
        return fetch_and( arg ).get_ctr() & arg;
    }
    int16_t operator|=( int16_t arg ) noexcept
    {
        return fetch_or( arg ).get_ctr() | arg;
    }
    int16_t operator^=( int16_t arg ) noexcept
    {
        return fetch_xor( arg ).get_ctr() ^ arg;
    }

private:
    std::atomic<uint64_t> word_;
    static_assert( std::atomic<uint64_t>::is_always_lock_free );
};


/*
 * Shared Pointers
 */

template<typename T> class shared_ptr;
template<typename T> class weak_ptr;
template<typename T> class atomic_shared_ptr;


template<typename T>
struct sptr_header_base {
public:
    /*
     * Increment the usage counter.
     */
    inline void acquire( std::memory_order order = std::memory_order_acquire ) noexcept
    {
        references_.fetch_add( { 0, 1 }, order );
    }
    inline void acquire( paired_counter count, std::memory_order order = std::memory_order_acquire ) noexcept
    {
        references_.fetch_add( count, order );
    }
    /*
     * Increases the counter for temporary observation.
     */
    void hold( int16_t count = 1, std::memory_order order = std::memory_order_acquire ) noexcept
    {
        references_.fetch_add( { count, 0 }, order );
    }
    void unhold( int16_t count = 1, std::memory_order order = std::memory_order_acquire ) noexcept
    {
        references_.fetch_sub( { count, 0 }, order );
    }
    /*
     * Decrement the usage counter, which might lead to the destruction of this object.
     */
    inline void release( paired_counter count = { 0, 1 }, std::memory_order order = std::memory_order_acquire ) noexcept
    {
        const auto old_ref = references_.fetch_sub( count, order );
        if( old_ref == count ) [[unlikely]] {
            _delete_object( get_ptr() );

            if( weak_references_.load( std::memory_order_acquire ).word() == 0 ) [[likely]]
                _delete_header();
        }
    }

    bool weak_lock( std::memory_order order = std::memory_order_acquire ) noexcept
    {
        auto cur_ref = references_.load( order );
        do {
            if( cur_ref.get_cnt2() == 0 )
                return false;
        } while( !references_.compare_exchange_weak( cur_ref,
                                                     paired_counter{
                                                             cur_ref.get_cnt1(),
                                                             cur_ref.get_cnt2()+1 },
                                                     std::memory_order_relaxed, std::memory_order_relaxed ));
        return true;
    }
    /*
     * Increment the weak usage counter.
     */
    inline void acquire_weak( std::memory_order order = std::memory_order_acquire ) noexcept
    {
        weak_references_.fetch_add( { 0, 1 }, order );
    }

    /*
     * Decrement the weak usage counter, which might lead to the destruction of the header (but never the object).
     */
    inline void release_weak( paired_counter count = { 0, 1 }, std::memory_order order = std::memory_order_acquire ) noexcept
    {
        const auto old_weak = weak_references_.fetch_sub( count, order );
        if( old_weak == paired_counter{ 0, 1 } && references_.load( std::memory_order_relaxed ).word() == 0 )
            _delete_header();
    }


    inline constexpr T* get_ptr() const noexcept
    {
        return pointer_;
    }
    inline uint32_t use_count() const noexcept
    {
        const auto ref = references_.load( std::memory_order_relaxed );
        return ref.get_cnt2();
    }
    inline uint32_t weak_count() const noexcept
    {
        const auto ref = weak_references_.load( std::memory_order_relaxed );
        return ref.get_cnt2();
    }
protected:
    constexpr explicit sptr_header_base( T* ptr, [[maybe_unused]] bool in_place = false ) noexcept
            : references_{{ 0, 1 }},
              weak_references_{ 0u },
              pointer_{ ptr }
    {}

    virtual ~sptr_header_base() noexcept
    {
        assert( references_.load( std::memory_order_acquire ).word() == 0 );
        assert( weak_references_.load( std::memory_order_acquire ).word() == 0 );
    }

    virtual void _delete_header() = 0;
    virtual void _delete_object( T* self ) = 0;

    /// temporary and global references
    atomic_paired_counter references_;

    /// weak counter references
    atomic_paired_counter weak_references_;

    /// the object pointer
    T* pointer_;
};


template<typename Allocator, bool is_deleter = false>
struct sptr_deleter {
    sptr_deleter( Allocator& allocator ) noexcept :
            allocator_{ allocator }
    {}

    template<typename Pointer>
    void delete_object( Pointer pointer )
    {
        if( is_deleter ) {
            auto deleter = allocator_;
            deleter( pointer );
        }
        else {
            auto alloc = allocator_;
            std::destroy_at( pointer );
            alloc.deallocate( pointer );
        }
    }

private:
    Allocator allocator_;
};
template<>
struct sptr_deleter<void> {
    template<typename Pointer>
    void delete_object( Pointer pointer )
    {
        delete pointer;
    }
};


template<typename T>
struct sptr_header_extern : public sptr_header_base<T> {
    constexpr explicit sptr_header_extern( T* p ) noexcept:
            sptr_header_base<T>{ p, false }
    {}

    ~sptr_header_extern()
    {
        assert( sptr_header_base<T>::references_.load( std::memory_order_acquire )
                == paired_counter( 0, 0 ) );
        assert( sptr_header_base<T>::weak_count() == 0 );
    }

    void _delete_header() override
    {
        delete this;
    }
    void _delete_object( T* self ) override
    {
        delete self;
    }
};


template<typename T, class Deleter>
struct sptr_header_extern_with_deleter : public sptr_header_base<T>,
                                         public sptr_deleter<Deleter, true> {

    sptr_header_extern_with_deleter( T* p, Deleter& deleter ) noexcept :
            sptr_header_base<T>{ p, false },
            sptr_deleter<Deleter, true>{ deleter }
    {}

    ~sptr_header_extern_with_deleter()
    {
        assert( sptr_header_base<T>::references_.load( std::memory_order_acquire )
                == paired_counter( 0, 0 ) );
        assert( sptr_header_base<T>::weak_count() == 0 );
    }

    void _delete_header() override
    {
        delete this;
    }
    void _delete_object( T* self ) override
    {
        sptr_deleter<Deleter, true>::delete_object( self );
    }
};


template<typename T, class Allocator>
struct sptr_header_extern_with_allocator : public sptr_header_base<T>,
                                           public sptr_deleter<Allocator> {

    sptr_header_extern_with_allocator( Allocator& allocator, T* p ) noexcept :
            sptr_header_base<T>{ p, false },
            sptr_deleter<Allocator>{ allocator }
    {}

    ~sptr_header_extern_with_allocator()
    {
        assert( sptr_header_base<T>::references_.load( std::memory_order_acquire )
                == paired_counter( 0, 0 ) );
        assert( sptr_header_base<T>::weak_count() == 0 );
    }

    void _delete_header() override
    {
        delete this;
    }
    void _delete_object( T* self ) override
    {
        sptr_deleter<Allocator>::delete_object( self );
    }
};


template<typename T>
struct sptr_header_inplace : public sptr_header_base<T> {
    template<typename... Args>
    explicit sptr_header_inplace( Args&&... args ) noexcept :
            sptr_header_base<T>{ reinterpret_cast<T*>( std::addressof( object_ )), true }
    {
        // construct the object
        std::construct_at( sptr_header_base<T>::get_ptr(), std::forward<Args>( args )... );
    }

    ~sptr_header_inplace() {
        assert( sptr_header_base<T>::references_.load( std::memory_order_acquire )
                == paired_counter( 0, 0 ) );
        assert( sptr_header_base<T>::weak_count() == 0 );
    }

    void _delete_object( T* self ) override
    {
        std::destroy_at( self );
    }

private:
    template<typename U, typename... Args> friend shared_ptr<U> make_shared( Args&&... argd );

    /// the object itself
    typename std::aligned_storage<sizeof( T ), alignof( T )>::type object_;
};


template<typename T, typename Deleter>
struct shareable : private sptr_header_base<T> {
    template<typename... Args>
    shareable( Deleter& deleter, Args... args ) :
            sptr_header_base<T>{ reinterpret_cast<T*>( std::addressof( object_ )), true },
            state_( live ),
            deleter_{ deleter }
    {
        //std::construct_at( sptr_header_base<T>::get_ptr(), std::forward<Args>( args )... );
        ::new( sptr_header_base<T>::get_ptr() ) T( std::forward<Args>( args )... );
    }

    T* operator->() noexcept
    {
        return sptr_header_base<T>::get_ptr();
    }
    const T* operator->() const noexcept
    {
        return sptr_header_base<T>::get_ptr();
    }
    T* operator*() noexcept
    {
        return sptr_header_base<T>::get_ptr();
    }
    const T* operator*() const noexcept
    {
        return sptr_header_base<T>::get_ptr();
    }

    operator shared_ptr<T>() noexcept
    {
        return shared_ptr<T>{ static_cast<sptr_header_base<T>*>( this ) };
    }

private:
    void _delete_object( [[maybe_unused]] T* self ) override {
        auto old_state = state_.fetch_or( destroying_object );
        assert( 0 == ( old_state & ( destroying_object | object_destroyed )));

        // call the destructor of the object
        std::destroy_at( sptr_header_base<T>::get_ptr() );

        // flip sptr_transition_state from destroying -> destroyed
        old_state = state_.fetch_xor( destroying_object | object_destroyed );
        assert( destroying_object == ( old_state & ( destroying_object | object_destroyed )));

        // maybe, we destroyed the object while another thread was faster and event wanted to destroy the header
        // now, this is our burden
        if( old_state & destroy_header ) {
            _delete_header();
        }
    }

    void _delete_header() override {
        // mark our intention to destroy the header
        auto old_state = state_.fetch_or( destroy_header );
        assert( old_state & ( destroying_object | object_destroyed ));

        // if destroying the object is not finished, we leave here, as the other thread will destroy the header for us
        if( old_state & destroying_object )
            return;

        // make a copy of the allocator_ before self-destruction
        auto deleter = deleter_;

        // explicitly destroy the header
        std::destroy_at( this );

        // deallocate the memory
        deleter( this );
    }

    static constexpr uint8_t live = 0;
    static constexpr uint8_t destroying_object = 1;
    static constexpr uint8_t object_destroyed = 2;
    static constexpr uint8_t destroy_header = 4;
    std::atomic<uint8_t> state_;

    Deleter deleter_;

    /// the object itself
    typename std::aligned_storage<sizeof( T ), alignof( T )>::type object_;

    template<typename U, class Alloc, typename... Args> friend shared_ptr<U> allocate_shared( Alloc&, Args&&... );
};


template<class T>
class shared_ptr {
private:
    using hdr_type = sptr_header_base<T>;
    using hdr_ptr_type = counted_ptr<hdr_type>;

    hdr_ptr_type cp_header_;

    template<class Y> friend class weak_ptr;
    template<class Y> friend class atomic_shared_ptr;
    template<class Y, class D> friend struct shareable;

    template<typename U, typename... Args>
    friend shared_ptr<U> make_shared( Args&&... );
    template<typename U, typename Alloc, typename... Args>
    friend shared_ptr<U> allocate_shared( Alloc&, Args&&... );

public:
    constexpr shared_ptr() noexcept : cp_header_{ 0, nullptr }
    {}
    constexpr shared_ptr( std::nullptr_t ) noexcept : cp_header_{ 0, nullptr }
    {}
    explicit shared_ptr( T* ptr ) : cp_header_{ 0, ptr? new sptr_header_extern{ ptr }:nullptr }
    {}
    template<class Allocator = std::allocator<T>>
    explicit shared_ptr( const Allocator& alloc, T* ptr ) :
            cp_header_{ 0, ptr ? new sptr_header_extern_with_allocator{ alloc, ptr }
                               : nullptr }
    {}
    template<class Deleter>
    explicit shared_ptr( T* ptr, const Deleter& deleter ) :
            cp_header_{ 0, ptr ? new sptr_header_extern_with_deleter{ ptr, deleter }
                               : nullptr }
    {}
    shared_ptr( shared_ptr&& r ) noexcept : cp_header_{ 0, nullptr }
    {
        swap( r );
    }
    shared_ptr( const shared_ptr& r ) noexcept : cp_header_{ 0, r.cp_header_.get_ptr() }
    {
        if( cp_header_.get_ptr() ) [[likely]]
            cp_header_->acquire();
    }
    ~shared_ptr()
    {
        if( cp_header_.get_ptr() ) [[likely]]
            cp_header_->release( { cp_header_.get_ctr(), 1 });
    }

private:
    constexpr explicit shared_ptr( sptr_header_base<T>* ctrl ) : cp_header_{ 0, ctrl }
    {}
    constexpr explicit shared_ptr( hdr_ptr_type ctrl ) : cp_header_{ ctrl }
    {}

public:
    shared_ptr& operator=( const shared_ptr& r ) noexcept {
        if( r.cp_header_.get_ptr() == cp_header_.get_ptr() )
            return *this;

        if( cp_header_.get_ptr() ) {
            cp_header_->release( { cp_header_.get_ctr(), 1 });
        }
        cp_header_ = { r.cp_header_.get_ptr() };
        if( cp_header_.get_ptr() )
            cp_header_->acquire();

        return *this;
    }
    shared_ptr& operator=( shared_ptr&& r ) noexcept {
        swap( r );
        //r.reset();
        return *this;
    }

    constexpr bool operator==( const shared_ptr& r ) const noexcept {
        return get() == r.get();
    }

    void reset() noexcept
    {
        _release();
        cp_header_ = { 0, nullptr };
    }
    void reset( T* ptr ) noexcept
    {
        _release();
        cp_header_ = { 0, ptr ? new sptr_header_extern{ ptr } : nullptr };
    }
    void swap( shared_ptr& r ) noexcept
    {
        std::swap( cp_header_, r.cp_header_ );
    }

    constexpr T* get() const noexcept
    {
        return cp_header_.get_ptr() ? cp_header_->get_ptr() : nullptr;
    }
    constexpr T& operator*() const noexcept
    {
        return *get();
    }
    constexpr T* operator->() const noexcept
    {
        return get();
    }
    [[nodiscard]] uint32_t use_count() const noexcept
    {
        return cp_header_.get_ptr() ? cp_header_->use_count() : 0;
    }
    uint32_t weak_count() const noexcept
    {
        return cp_header_.get_ptr() ? cp_header_->weak_count() : 0;
    }
    [[nodiscard]] bool unique() const noexcept
    {
        return use_count() == 1;
    }
    explicit operator bool() const noexcept
    {
        return use_count() != 0;
    }

private:

    void _release() noexcept
    {
        if( cp_header_.get_ptr() ) [[likely]]
            cp_header_->release( { cp_header_.get_ctr(), 1 }, std::memory_order_acquire );
    }
    void _acquire() const noexcept
    {
        if( cp_header_.get_ptr() ) [[likely]]
            cp_header_->acquire( std::memory_order_relaxed );
    }
};


template<typename T>
class weak_ptr {
private:
    using hdr_type = typename shared_ptr<T>::hdr_type;
    using hdr_ptr_type = typename shared_ptr<T>::hdr_ptr_type;

    hdr_ptr_type cp_header_;

public:
    constexpr weak_ptr() noexcept : cp_header_{ 0, nullptr }
    {}
    constexpr weak_ptr( std::nullptr_t ) noexcept : cp_header_{ 0, nullptr }
    {}
    weak_ptr( const weak_ptr& r ) noexcept : cp_header_{ 0, r.cp_header_.get_ptr() }
    {
        if( cp_header_.get_ptr() )
            cp_header_->acquire_weak( std::memory_order_relaxed );
    }
    weak_ptr( const shared_ptr<T>& r ) noexcept : cp_header_{ 0, r.cp_header_.get_ptr() }
    {
        if( cp_header_.get_ptr() )
            cp_header_->acquire_weak( std::memory_order_relaxed );
    }
    weak_ptr( weak_ptr&& r ) noexcept : cp_header_{ 0, nullptr }
    {
        swap( r );
    }

    ~weak_ptr()
    {
        if( cp_header_.get_ptr() )
            cp_header_->release_weak( { cp_header_.get_ctr(), 1 }, std::memory_order_acquire );
    }

    weak_ptr& operator=( const weak_ptr& r )
    {
        if( cp_header_.get_ptr() )
            cp_header_->release_weak( { cp_header_.get_ctr(), 1 }, std::memory_order_acquire );
        cp_header_ = { 0, r.cp_header_.get_ptr() };
        if( cp_header_.get_ptr() )
            cp_header_->acquire_weak( std::memory_order_relaxed );
        return *this;
    }
    weak_ptr& operator=( const shared_ptr<T>& r )
    {
        if( cp_header_.get_ptr() )
            cp_header_->release_weak( { cp_header_.get_ctr(), 1 }, std::memory_order_acquire );
        cp_header_ = { 0, r.cp_header_.get_ptr() };
        if( cp_header_.get_ptr() )
            cp_header_->acquire_weak( std::memory_order_relaxed );
    }
    weak_ptr& operator=( weak_ptr&& r )
    {
        swap( r );
    }

    void reset()
    {
        if( cp_header_.get_ptr() ) {
            cp_header_->release_weak( { -cp_header_.get_ctr(), 1 }, std::memory_order_acquire );
            cp_header_ = { 0, nullptr };
        }
    }
    void swap( weak_ptr& r ) noexcept
    {
        std::swap( cp_header_, r.cp_header_ );
    }

    uint32_t use_count() const noexcept
    {
        return cp_header_.get_ptr() ? cp_header_->use_count() : 0;
    }
    uint32_t weak_count() const noexcept
    {
        return cp_header_.get_ptr() ? cp_header_->weak_count() : 0;
    }
    bool expired() const noexcept
    {
        return use_count() == 0;
    }
    shared_ptr<T> lock() const noexcept
    {
        if( cp_header_.get_ptr() == nullptr )
            return {};

        if( !cp_header_->weak_lock( std::memory_order_acquire ))
            return {};

        return shared_ptr<T>{ cp_header_.get_ptr() };
    }
    bool owner_before( const weak_ptr& r ) const noexcept
    {
        return cp_header_.get_ptr() < r.cp_header_.get_ptr();
    }
    bool owner_before( const shared_ptr<T>& r ) const noexcept
    {
        return cp_header_.get_ptr() < r.cp_header_.get_ptr();
    }
};


template<typename T>
class alignas( CACHE_COHERENCY_LINE_SIZE ) atomic_shared_ptr {
private:
    using hdr_type = sptr_header_base<T>;
    using hdr_ptr_type = counted_ptr<hdr_type>;

    mutable atomic_counted_ptr<hdr_type> cptr_hdr_;

public:
    constexpr static bool is_always_lock_free = atomic_counted_ptr<hdr_type>::is_always_lock_free;

    constexpr atomic_shared_ptr() noexcept : cptr_hdr_{ 0, nullptr }
    {}
    constexpr atomic_shared_ptr( std::nullptr_t ) noexcept : cptr_hdr_{ 0, nullptr }
    {}
    explicit atomic_shared_ptr( T* ptr ) : cptr_hdr_{ 0, new sptr_header_extern<T>{ ptr } }
    {}
    atomic_shared_ptr( shared_ptr<T>& r ) noexcept : cptr_hdr_{ 0, r.cp_header_.get_ptr() }
    {
        r._acquire();
    }
    atomic_shared_ptr( shared_ptr<T>&& r ) noexcept : cptr_hdr_{ r.cp_header_ }
    {
        r.cp_header_ = { 0, nullptr };
    }
    ~atomic_shared_ptr()
    {
        const auto cur_ctrl_ptr = cptr_hdr_.load( std::memory_order_acquire );

        // fix negative local counter situations due to ABA problems
        if( cur_ctrl_ptr.get_ptr() )
            cur_ctrl_ptr.get_ptr()->release( { cur_ctrl_ptr.get_ctr(), 1 }, std::memory_order_acquire );
    }

    atomic_shared_ptr& operator=( const shared_ptr<T>& r ) noexcept
    {
        store( r );
        return *this;
    }
    atomic_shared_ptr& operator=( std::nullptr_t ) noexcept
    {
        store({ 0, nullptr });
        return *this;
    }

    operator shared_ptr<T>() const noexcept
    {
        return load( std::memory_order_acquire );
    }

    bool is_lock_free() const noexcept
    {
        return cptr_hdr_.is_lock_free();
    }

    inline void store( const shared_ptr<T>& desired, std::memory_order order = std::memory_order_seq_cst ) noexcept {
        store( shared_ptr<T>{ desired }, order );
    }
    inline void store( shared_ptr<T>&& desired, std::memory_order order = std::memory_order_seq_cst ) noexcept {
        desired.cp_header_ = cptr_hdr_.exchange( desired.cp_header_, order );
    }
    shared_ptr<T> load( std::memory_order order = std::memory_order_seq_cst ) const noexcept {
        // increment local ref counter, simultaneously reading the object
        auto cur_ctrl_ptr = _enter( std::memory_order_relaxed );
        if( cur_ctrl_ptr.get_ptr() == nullptr ) [[unlikely]]
            return shared_ptr<T>{ nullptr };

        cur_ctrl_ptr->acquire( { 1, 1 }, order );

        return shared_ptr<T>{ cur_ctrl_ptr.get_ptr() };
    }
    shared_ptr<T> exchange( const shared_ptr<T>& desired, std::memory_order order = std::memory_order_seq_cst ) noexcept
    {
        const auto desired_ptr = desired.cp_header_.get_ptr();
        if( desired_ptr )
            desired_ptr->acquire({ 0, 1 });
        return shared_ptr<T>{ cptr_hdr_.exchange( hdr_ptr_type{ desired_ptr }, order ) };
    }
    shared_ptr<T> exchange( shared_ptr<T>&& desired, std::memory_order order = std::memory_order_seq_cst ) noexcept
    {
        shared_ptr<T> old_sptr{ cptr_hdr_.exchange( desired.cp_header_, order ) };
        desired.cp_header_ = { 0, nullptr };
        return old_sptr;
    }


    bool compare_exchange_weak( shared_ptr<T>& expected, shared_ptr<T>&& desired,
                                std::memory_order success, std::memory_order failure ) noexcept
    {
        const auto expected_ptr = expected.cp_header_.get_ptr();
        hdr_ptr_type exp_ctrl_ptr;
        goto start;

        for(;;) {
            // Do an optimistic cas
            if( cptr_hdr_.compare_exchange_weak( exp_ctrl_ptr, desired.cp_header_, success, failure )) {
                desired.cp_header_ = exp_ctrl_ptr;
                return true;
            }
            if( expected_ptr != exp_ctrl_ptr.get_ptr() ) {
start:
                exp_ctrl_ptr = _enter( std::memory_order_relaxed );
                if( expected_ptr != exp_ctrl_ptr.get_ptr() ) [[likely]] {
                    expected = shared_ptr<T>{ exp_ctrl_ptr.get_ptr() };
                    if( exp_ctrl_ptr.get_ptr() )
                        exp_ctrl_ptr->acquire( { 1, 1 }, std::memory_order_relaxed );
                    return false;
                }

                expected.cp_header_.counter()--;  // compensate the _enter() from above
            }
        }
    }
    bool compare_exchange_weak( shared_ptr<T>& expected, const shared_ptr<T>& desired,
                                std::memory_order success, std::memory_order failure ) noexcept
    {
        {
            const hdr_ptr_type desired_cptr = { desired.cp_header_.get_ptr() };
            const auto expected_ptr = expected.cp_header_.get_ptr();
            hdr_ptr_type exp_ctrl_ptr; //{ expected_ptr };
            bool acquired_des = false;
            goto start;

            for(;;) {
                // Do an optimistic cas
                if( cptr_hdr_.compare_exchange_weak( exp_ctrl_ptr, desired_cptr, success, failure )) {
                    if( expected_ptr )
                        expected_ptr->release( { exp_ctrl_ptr.get_ctr(), 1 }, std::memory_order_relaxed );
                    return true;
                }
                if( expected_ptr != exp_ctrl_ptr.get_ptr() ) {
start:
                    exp_ctrl_ptr = _enter( std::memory_order_relaxed );
                    if( expected_ptr != exp_ctrl_ptr.get_ptr() ) [[likely]] {
                        if( acquired_des ) {
                            if( desired_cptr.get_ptr()) [[likely]]
                                desired_cptr->release( { 0, 1 }, std::memory_order::relaxed );
                        }

                        if( exp_ctrl_ptr.get_ptr() ) [[likely]]
                            exp_ctrl_ptr->acquire( { 1, 1 }, std::memory_order_relaxed );
                        expected = shared_ptr<T>{ exp_ctrl_ptr.get_ptr() };
                        return false;
                    }

                    expected.cp_header_.counter()--;  // compensate the _enter() from above

                    // acquire new instance of *desired* before the cas might succeed!
                    if( !acquired_des ) {
                        if( desired_cptr.get_ptr()) [[likely]]
                            desired_cptr->acquire( { 0, 1 }, std::memory_order::relaxed );
                        acquired_des = true;
                    }
                }
            }
        }
    }

    bool compare_exchange_weak( shared_ptr<T>& expected, const shared_ptr<T>& desired,
                                std::memory_order order = std::memory_order_seq_cst ) noexcept
    {
        return compare_exchange_weak( expected, desired, order, order );
    }
    bool compare_exchange_weak( shared_ptr<T>& expected, shared_ptr<T>&& desired,
                                std::memory_order order = std::memory_order_seq_cst ) noexcept
    {
        return compare_exchange_weak( expected, std::move( desired ), order, order );
    }


    bool compare_exchange_strong( shared_ptr<T>& expected, shared_ptr<T>&& desired,
                                  std::memory_order success, std::memory_order failure ) noexcept
    {
        /*
         * The cas does an optimistic approach: it does the cas without holding on to the object. If it succeeds,
         * the object will be valid since we have it acquired through the expected variable. It, however, the cas
         * fails, the hard part is to get a hold onto the old object:
         * Only on failure (because of a non-expected pointer), we enter_() the expected variable. By then, it could
         * have been changed - enter gets a hold to the changed value, which we can then materialize to the expected
         * variable. However, it can also have been changed back to our expected variable. In this case, we can try
         * the cas again.
         */
        const auto expected_ptr = expected.cp_header_.get_ptr();
        hdr_ptr_type exp_ctrl_ptr;
        goto start;

        for(;;) {
            // Do an optimistic cas
            if( cptr_hdr_.compare_exchange_strong( exp_ctrl_ptr, desired.cp_header_, success, failure )) {
                desired.cp_header_ = exp_ctrl_ptr;
                return true;
            }
            if( expected_ptr != exp_ctrl_ptr.get_ptr() ) [[likely]] {
start:
                exp_ctrl_ptr = _enter( std::memory_order_relaxed );
                if( expected_ptr != exp_ctrl_ptr.get_ptr() ) [[likely]] {
                    expected = shared_ptr<T>{ exp_ctrl_ptr.get_ptr() };
                    if( exp_ctrl_ptr.get_ptr() ) [[likely]]
                        exp_ctrl_ptr.get_ptr()->acquire( { 1, 1 }, std::memory_order_relaxed );
                    return false;
                }
                else
                    expected.cp_header_.counter()--;  // compensate the _enter() from above
            }
        }
    }
    bool compare_exchange_strong( shared_ptr<T>& expected, const shared_ptr<T>& desired,
                                  std::memory_order success, std::memory_order failure ) noexcept
    {
        const hdr_ptr_type desired_cptr = { desired.cp_header_.get_ptr() };
        const auto expected_ptr = expected.cp_header_.get_ptr();
        hdr_ptr_type exp_ctrl_ptr;
        bool acquired_des = false;
        goto start;

        for(;;) {
            // Do an optimistic cas
            if( cptr_hdr_.compare_exchange_strong( exp_ctrl_ptr, desired_cptr, success, failure )) {
                if( expected_ptr )
                    expected_ptr->release({ exp_ctrl_ptr.get_ctr(), 1 }, std::memory_order_relaxed );
                return true;
            }
            if( expected_ptr != exp_ctrl_ptr.get_ptr() ) {
start:
                exp_ctrl_ptr = _enter( std::memory_order_relaxed );
                if( expected_ptr != exp_ctrl_ptr.get_ptr() ) [[likely]] {
                    if( acquired_des ) {
                        if( desired_cptr.get_ptr()) [[likely]]
                            desired_cptr->release( { 0, 1 }, std::memory_order::relaxed );
                    }

                    if( exp_ctrl_ptr.get_ptr() ) [[likely]]
                        exp_ctrl_ptr.get_ptr()->acquire( { 1, 1 }, std::memory_order_relaxed );
                    expected = shared_ptr<T>{ exp_ctrl_ptr.get_ptr() };
                    return false;
                }

                expected.cp_header_.counter()--;  // compensate the _enter() from above

                // acquire new instance of *desired* before the cas might succeed!
                if( !acquired_des ) {
                    if( desired_cptr.get_ptr()) [[likely]]
                        desired_cptr->acquire( { 0, 1 }, std::memory_order::relaxed );
                    acquired_des = true;
                }
            }
        }
    }
    bool compare_exchange_strong( shared_ptr<T>& expected, const shared_ptr<T>& desired,
                                  std::memory_order order = std::memory_order_seq_cst ) noexcept
    {
        return compare_exchange_strong( expected, desired, order, order );
    }
    bool compare_exchange_strong( shared_ptr<T>& expected, shared_ptr<T>&& desired,
                                  std::memory_order order = std::memory_order_seq_cst ) noexcept
    {
        return compare_exchange_strong( expected, std::move( desired ), order, order );
    }

    void wait( shared_ptr<T> old, std::memory_order order = std::memory_order_seq_cst ) noexcept
    {
        const auto cur_ctrl = _enter( order );
        for(;;) {
            if( cur_ctrl.get_ptr() == old.get() )
                cptr_hdr_.wait( cur_ctrl );
            else {
                _leave( cur_ctrl, std::memory_order_relaxed );
                return;
            }
            cur_ctrl = _reenter( cur_ctrl, std::memory_order_relaxed );
        }
    }
    void notify_one() noexcept
    {
        cptr_hdr_.notify_one();
    }
    void notify_all() noexcept
    {
        cptr_hdr_.notify_all();
    }

private:

    /*
     * Decreases a global ref count, possibly deleting the object and the control block.
     */
    static void _release( hdr_type* ctrl_ptr, paired_counter count = { 0, 1 },
                          std::memory_order order = std::memory_order_acq_rel ) noexcept
    {
        if( ctrl_ptr ) [[likely]]
            ctrl_ptr->release( count, order );
    }

    /*
     * Increases the local ref counter and returns the new local ref counter and pointer to the control block.
     */
    hdr_ptr_type _enter( std::memory_order order = std::memory_order_relaxed ) const noexcept
    {
        auto ctrl_ptr = cptr_hdr_.fetch_add( 1, order );
        ctrl_ptr.counter()++;

        // normalize
        if( ctrl_ptr.get_ctr() >= 1 << 14 && ctrl_ptr.get_ptr() ) [[unlikely]] {
            if( _try_leave( ctrl_ptr, ctrl_ptr.get_ctr())) {
                ctrl_ptr.set_ctr( 0 );
                ctrl_ptr->unhold( ctrl_ptr.get_ctr(), std::memory_order_relaxed );
            }
        }

        return ctrl_ptr;
    }
    /*
     * Decreases the local ref count. If, however, the ptr has been reassigned in the meantime,
     * the local ref count increment has been transferred to the global ref count, in which case the global ref
     * count will be decremented, possibly deleting the object and control block.
     */
    void _leave( hdr_ptr_type cur_ctrl_ptr,
                 std::memory_order order = std::memory_order_relaxed ) const noexcept
    {
        // reduce the local ref count by one (or the global ref count if there was a reassignment)
        for(;;) {
            //assert( cur_ctrl_ptr.get_ptr() == nullptr || cur_ctrl_ptr.get_ctr() >= 1 );

            hdr_ptr_type desired_ctrl_ptr{ int16_t( cur_ctrl_ptr.get_ctr() - 1 ),
                                           cur_ctrl_ptr.get_ptr() };
            if( cptr_hdr_.compare_exchange_weak( cur_ctrl_ptr, desired_ctrl_ptr, order ))
                return;

            if( cur_ctrl_ptr.get_ptr() != desired_ctrl_ptr.get_ptr() ) {
                _release( desired_ctrl_ptr.get_ptr(), { 1, 0 } );
                return;
            }
        }
    }
    bool _try_leave( hdr_ptr_type cur_ctrl_ptr, int16_t count,
                 std::memory_order order = std::memory_order_relaxed ) const noexcept
    {
        const hdr_ptr_type desired_ctrl_ptr{ int16_t( cur_ctrl_ptr.get_ctr() - count ),
                                             cur_ctrl_ptr.get_ptr() };
        return cptr_hdr_.compare_exchange_strong( cur_ctrl_ptr, desired_ctrl_ptr, order );
    }
    hdr_ptr_type _reenter( hdr_ptr_type old_ctrl_ptr,
                           std::memory_order order = std::memory_order_relaxed ) const noexcept
    {
        auto cur_ctrl_ptr = cptr_hdr_.load( std::memory_order_relaxed );
        if( cur_ctrl_ptr.get_ptr() == old_ctrl_ptr.get_ptr() )
            return cur_ctrl_ptr;

        if( old_ctrl_ptr.get_ptr() )
            old_ctrl_ptr->release({ 1, 0 }, order );
        return _enter( order );
    }
};


template<typename T, typename... Args>
shared_ptr<T> make_shared( Args&&... args )
{
    struct hdr_default_alloc : public sptr_header_inplace<T> {
        hdr_default_alloc( Args&&... args ) : sptr_header_inplace<T>{ std::forward<Args>( args )... }
        {}
        void _delete_header() override
        {
            delete this;
        }
    };

    return shared_ptr<T>{ new hdr_default_alloc{ std::forward<Args>( args )... }};
}

template<typename T, class Alloc, typename... Args>
shared_ptr<T> allocate_shared( Alloc& alloc, Args&&... args ) {
    auto* hdr = std::allocator_traits<Alloc>::allocate( alloc, 1 );
    std::allocator_traits<Alloc>::construct( alloc, hdr, std::forward<Args>( args )... );
    return shared_ptr<T>{ hdr };
}

}
