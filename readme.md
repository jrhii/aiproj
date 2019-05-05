# Notes of black friday csv
Data was cleaned and digitized using a regex plugin.
empty entries in the product category columns represent a product not purchased, and were changed to `0`
Gender was mapped as follows
    `F -> 0`
    `M -> 1`
Age groups were mapped as follows
    `0-17  -> 0`
    `18-25 -> 1`
    `26-35 -> 2`
    `36-45 -> 3`
    `46-50 -> 4`
    `51-55 -> 5`
    `55+   -> 6`
City category was mapped as follows
    `A -> 0`
    `B -> 1`
    `C -> 2`
For stay in current years, `4+` was changed to `4`
`P` was removed from every product id, as the numbers are still unique without the P prefix that they all share