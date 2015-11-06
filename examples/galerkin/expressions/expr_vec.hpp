/**@file vector of accessors

   used in order to implement vector algebra
*/

namepace gridtools{

    template <typename ... Accessors>
        struct expr_vec() : public expr<Accessors ... > {

    }

}
