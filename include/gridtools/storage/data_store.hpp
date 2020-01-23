/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <memory>
#include <string>
#include <type_traits>

#include "../common/array.hpp"
#include "../common/array_addons.hpp"
#include "../common/defs.hpp"
#include "../common/layout_map.hpp"
#include "data_view.hpp"
#include "info.hpp"
#include "traits.hpp"

namespace gridtools {
    namespace storage {

        struct uninitialized {};

        namespace data_store_impl_ {

            inline constexpr int gcd(size_t a, size_t b) {
                if (b == 0)
                    return a;
                return gcd(b, a % b);
            }

            template <class Traits, class T, size_t N, class Id>
            class base {
                static constexpr size_t byte_alignment = traits::alignment<Traits>;
                static_assert(byte_alignment > 0, GT_INTERNAL_ERROR);

                static constexpr size_t alignment = byte_alignment / gcd(sizeof(T), byte_alignment);

                using mutable_data_t = std::remove_const_t<T>;

                std::string m_name;
                storage::info<N> m_info;
                traits::target_ptr_type<Traits, mutable_data_t> m_target_ptr_holder;
                mutable_data_t *m_target_ptr;

              public:
                using layout_t = traits::layout_type<Traits, N>;
                using data_t = T;
                static constexpr size_t ndims = N;

                using kind_t = meta::list<layout_t,
                    meta::if_c<(layout_t::unmasked_length > 0), Id, void>,
                    meta::if_c<(layout_t::unmasked_length > 1), std::integral_constant<size_t, alignment>, void>>;

                auto const &name() const { return m_name; }
                auto const &info() const { return m_info; }
                auto const &lengths() const { return m_info.lengths(); }
                auto const &strides() const { return m_info.strides(); }
                auto length() const { return m_info.length(); }

              protected:
                base(std::string name, array<uint_t, N> const &lengths, array<int, N> const &halos)
                    : m_name(std::move(name)), m_info(layout_t(), alignment, lengths),
                      m_target_ptr_holder(traits::allocate<Traits, mutable_data_t>(m_info.length() + alignment)) {
                    auto offset_to_align = m_info.index(halos);
                    auto byte_offset = offset_to_align * sizeof(T);
                    auto address_to_align = reinterpret_cast<std::uintptr_t>(m_target_ptr_holder.get()) + byte_offset;
                    m_target_ptr = reinterpret_cast<mutable_data_t *>(
                        (address_to_align + byte_alignment - 1) / byte_alignment * byte_alignment - byte_offset);
                }

                auto raw_target_ptr() const { return m_target_ptr; }
            };

            template <class Traits,
                class T,
                size_t N,
                class Id,
                bool = std::is_const<T>::value,
                bool = traits::is_host_referenceable<Traits>>
            class data_store_impl;

            template <class Traits, class T, size_t N, class Id>
            class data_store_impl<Traits, T, N, Id, false, false> : public base<Traits, T, N, Id> {
                enum state { synced, invalid_host, invalid_target };
                state m_state;
                std::unique_ptr<T[]> m_host_ptr;

                void update_target() {
                    if (m_state != invalid_target)
                        return;
                    traits::update_target<Traits>(this->raw_target_ptr(), m_host_ptr.get(), this->info().length());
                    m_state = synced;
                }

                void update_host() {
                    if (m_state != invalid_host)
                        return;
                    traits::update_host<Traits>(m_host_ptr.get(), this->raw_target_ptr(), this->info().length());
                    m_state = synced;
                }

              public:
                data_store_impl(std::string name,
                    array<uint_t, N> const &lengths,
                    array<int, N> const &halos,
                    uninitialized const &)
                    : data_store_impl::base(std::move(name), lengths, halos), m_state(synced),
                      m_host_ptr(std::make_unique<T[]>(this->info().length())) {}

                template <class Initializer>
                data_store_impl(std::string name,
                    array<uint_t, N> const &lengths,
                    array<int, N> const &halos,
                    Initializer const &initializer)
                    : data_store_impl::base(std::move(name), lengths, halos), m_state(invalid_target),
                      m_host_ptr(std::make_unique<T[]>(this->info().length())) {
                    initializer(m_host_ptr.get(), typename data_store_impl::layout_t(), this->info());
                }

                T *get_target_ptr() {
                    update_target();
                    m_state = invalid_host;
                    return this->raw_target_ptr();
                }

                T const *get_const_target_ptr() {
                    update_target();
                    return this->raw_target_ptr();
                }

                T *get_host_ptr() {
                    update_host();
                    m_state = invalid_target;
                    return m_host_ptr.get();
                }

                T const *get_const_host_ptr() {
                    update_host();
                    return m_host_ptr.get();
                }

                auto host_view() { return make_host_view(get_host_ptr(), this->info()); }
                auto const_host_view() { return make_host_view(get_const_host_ptr(), this->info()); }

                auto target_view() {
                    return traits::make_target_view<Traits, typename data_store_impl::kind_t>(
                        get_target_ptr(), this->info());
                }
                auto const_target_view() {
                    return traits::make_target_view<Traits, typename data_store_impl::kind_t>(
                        get_const_target_ptr(), this->info());
                }
            };

            template <class Traits, class T, size_t N, class Id>
            class data_store_impl<Traits, T, N, Id, false, true> : public base<Traits, T, N, Id> {
              public:
                data_store_impl(std::string name,
                    array<uint_t, N> const &lengths,
                    array<int, N> const &halos,
                    uninitialized const &)
                    : data_store_impl::base(std::move(name), lengths, halos) {}

                template <class Initializer>
                data_store_impl(std::string name,
                    array<uint_t, N> const &lengths,
                    array<int, N> const &halos,
                    Initializer const &initializer)
                    : data_store_impl::base(std::move(name), lengths, halos) {
                    initializer(this->raw_target_ptr(), typename data_store_impl::layout_t(), this->info());
                }

                T *get_target_ptr() const { return this->raw_target_ptr(); }
                T const *get_const_target_ptr() const { return this->raw_target_ptr(); }

                auto target_view() const {
                    return traits::make_target_view<Traits, typename data_store_impl::kind_t>(
                        get_target_ptr(), this->info());
                }
                auto const_target_view() const {
                    return traits::make_target_view<Traits, typename data_store_impl::kind_t>(
                        get_const_target_ptr(), this->info());
                }

                T *get_host_ptr() { return get_target_ptr(); }
                T const *get_const_host_ptr() { return get_const_target_ptr(); }
                auto host_view() const { return target_view(); }
                auto const_host_view() const { return const_target_view(); }
            };

            template <class Traits, class T, size_t N, class Id, bool IsHostRefrenceable>
            class data_store_impl<Traits, T const, N, Id, true, IsHostRefrenceable>
                : public base<Traits, T const, N, Id> {

                template <class>
                struct is_host_refrenceable : bool_constant<IsHostRefrenceable> {};

                template <class Initializer, std::enable_if_t<!is_host_refrenceable<Initializer>::value, int> = 0>
                void init(Initializer const &initializer) {
                    auto host_ptr = std::make_unique<T[]>(this->info().length());
                    initializer(host_ptr.get(), typename data_store_impl::layout_t(), this->info());
                    traits::update_target<Traits>(this->raw_target_ptr(), host_ptr.get(), this->info().length());
                }

                template <class Initializer, std::enable_if_t<is_host_refrenceable<Initializer>::value, int> = 0>
                void init(Initializer const &initializer) {
                    initializer(this->raw_target_ptr(), typename data_store_impl::layout_t(), this->info());
                }

              public:
                data_store_impl(std::string name,
                    array<uint_t, N> const &lengths,
                    array<int, N> const &halos,
                    uninitialized const &) = delete;

                template <class Initializer>
                data_store_impl(std::string name,
                    array<uint_t, N> const &lengths,
                    array<int, N> const &halos,
                    Initializer const &initializer)
                    : base<Traits, T const, N, Id>(std::move(name), lengths, halos) {
                    init(initializer);
                }
                T const *get_target_ptr() const { return this->raw_target_ptr(); }
                auto target_view() const {
                    return traits::make_target_view<Traits, typename data_store_impl::kind_t>(
                        get_target_ptr(), this->info());
                }
                auto get_const_target_ptr() const { return get_target_ptr(); }
                auto const_target_view() const { return target_view(); }
            };
        } // namespace data_store_impl_

        template <class Traits, class T, size_t N, class Id>
        struct data_store : data_store_impl_::data_store_impl<Traits, T, N, Id> {
            using data_store_impl_::data_store_impl<Traits, T, N, Id>::data_store_impl;
        };

        template <class>
        struct is_data_store : std::false_type {};

        template <class Traits, class T, size_t N, class Id>
        struct is_data_store<data_store<Traits, T, N, Id>> : std::true_type {};

        template <class>
        struct is_data_store_ptr : std::false_type {};

        template <class Traits, class T, size_t N, class Id>
        struct is_data_store_ptr<std::shared_ptr<data_store<Traits, T, N, Id>>> : std::true_type {};

        template <class Traits, class T, class Id, size_t N, class Initializer>
        auto make_data_store(std::string name,
            array<uint_t, N> const &lengths,
            array<int, N> const &halos,
            Initializer const &initializer) {
            return std::make_shared<data_store<Traits, T, N, Id>>(std::move(name), lengths, halos, initializer);
        }
    } // namespace storage
} // namespace gridtools
