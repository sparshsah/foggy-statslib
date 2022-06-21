author: [@sparshsah](https://github.com/sparshsah)

# Package structure

* `core`
* `stat`istic`s`
    - `sim`ulations
    - `est`imation
    - `t`ime`s`eries `a`nalysis
* `fin`ance


# Style notes

* We permit ourselves our personal predilection for underscores, to the point of controversy and perhaps even overuse.
  For example, if we have a 2D array `arr`, we will iterate over each row within that array as `_arr`.
  Similarly, if we have a function `foo`, we will call its helper `_foo`, and in turn its helper `__foo`.
  This nomenclature takes a little getting-used-to, but we vigorously defend the modularity and clarity it promotes.
  For example, building a nested series of one-liner helpers becomes second nature, so that
  each individual function is easy to digest and its position in the hierarchy is immediately obvious.
  In fact, if you think of a row within an array (or a helper to a function) as a "private" property,
  you might even call this at-first-glance unidiomatic nomenclature truly Pythonic!
