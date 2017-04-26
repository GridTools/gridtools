#### Conditionals

Conditionals introduce two new syntactic elements in the
[Grammar](computation grammar), namely ```if_``` and ```switch_```.
These implement run-time branches in the computation tree, i.e. one computation
or another can be chosen based on the value of a runtime variable.
Note that this is just syntactic sugar, as you could instantiate all the
possible combinations of the computation tree and then choose which one to
execute by querying the value of a runtime condition. This would work
in the same way as the solution we will describe next, but it would create a
code of an unmanageable size most of the time: suppose that inside a computation
with 10 stages you want to choose among 5 possible versions of the last stage.
You would have to create 5 different computations, in which the only difference
is in the last stage, while the rest is repeated. This would be tedious and error-prone.

The syntax we expose for ```if_``` statements is reported in the following example

```c++
   auto cond = new_cond([&flag]() { return flag; });
   comp_ = make_computation< BACKEND >(
           domain_,
           grid_,
           if_(cond,
               make_multistage(enumtype::execute< enumtype::forward >(), make_stage< functor0 >(p())),
               make_multistage(enumtype::execute< enumtype::forward >(), make_stage< functor1 >(p()))));
```
In this example ```cond``` is defined as a predicate using the ```new_cond``` $\GT$ keyword. Note that this code will always only
be executed on the host, so the predicate can access values which are available on the host, so captures by reference can be used.
The correct way to interpret this syntax is that the call to ```if_``` returns one multistage o the other based on
the return value of the predicate attached to ```cond```.

The value of ```cond``` is evaluated at every execution of the ```comp_.run()``` function.

The conditionals can also be nested

```c++
   auto cond = new_cond([]() { return false; });
   auto cond2 = new_cond([]() { return true; });
   comp_ = make_computation< BACKEND >(
           domain_,
           grid_,
           if_(cond,
               make_multistage(enumtype::execute< enumtype::forward >(), make_stage< functor0 >(p())),
               if_(cond2,
                   make_multistage(
                       enumtype::execute< enumtype::forward >(), make_stage< functor1 >(p())),
                   make_multistage(
                       enumtype::execute< enumtype::forward >(), make_stage< functor2 >(p())))));
```

The other syntactic element we introduce is a ```switch_```,
and its use is exemplified in the following snippet
```c++
   auto cond_ = new_switch_variable([&p]() { return p ? 0 : 5; });
   auto comp_ = make_computation< BACKEND >(
       domain_,
       grid_,
       make_multistage(enumtype::execute< enumtype::forward >(),
           make_stage< functor0 >(p(), p_tmp()),
           make_stage< functor1 >(p(), p_tmp())),
       switch_(cond_,
           case_(0,
                   make_multistage(enumtype::execute< enumtype::forward >(),
                       make_stage< functor1 >(p(), p_tmp()),
                       make_stage< functor2 >(p(), p_tmp()))),
           case_(1,
                   make_multistage(enumtype::execute< enumtype::forward >(),
                       make_stage< functor1 >(p(), p_tmp()),
                       make_stage< functor2 >(p(), p_tmp()))),
           case_(2,
                   make_multistage(enumtype::execute< enumtype::forward >(),
                       make_stage< functor1 >(p(), p_tmp()),
                       make_stage< functor2 >(p(), p_tmp()))),
           case_(3,
                   make_multistage(enumtype::execute< enumtype::forward >(),
                       make_stage< functor1 >(p(), p_tmp()),
                       make_stage< functor2 >(p(), p_tmp()))),
           case_(4,
                   make_multistage(enumtype::execute< enumtype::forward >(),
                       make_stage< functor1 >(p(), p_tmp()),
                       make_stage< functor2 >(p(), p_tmp())))))
```

As for the ```if_``` statement, ```cond_``` is evaluated at every call to ```comp_->run()```, and the
multistage executed in the two calls will be different.

---------------------------------------------------   --------------------------------------------------------
![Tip](figures/hint.gif){ width=20px height=20px }                                                        
                                                      Also ```switch_``` can be nested, as the ```if_```.
---------------------------------------------------   --------------------------------------------------------


---------------------------------------------------   --------------------------------------------------------
![Tip](figures/hint.gif){ width=20px height=20px }                                                        
                                                      The effect of having different branches is that all the
                                                      possibilities get compiled, and only one gets chosen at
                                                      each run. Therefore having lot of branches can increase
                                                      dramatically the compilation times, you should not abuse
                                                      of this feature.
---------------------------------------------------   --------------------------------------------------------

---------------------------------------------------   --------------------------------------------------------
![Tip](figures/hint.gif){ width=20px height=20px }                                                        
                                                      Currently there is a limitation. The different branches
                                                      in the computation must use the same placeholders.
---------------------------------------------------   --------------------------------------------------------
