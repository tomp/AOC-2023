# AOC-2023
Python solutions for the 2023 Advent of Code http://adventofcode.com/2023


Overview
--------
This repo contains my solutions to the 2023 challenges, and some helper code
that I use to get started, each day.

#### new_day.sh
Every day, after that day's puzzle has come online, I execute the
`new_day.sh` script and a new subfolder is created for that day's puzzle.
This saves me a few minutes of typing boilerplate code and downloading input
data, and let's me dive right into the puzzle.

For example, on day 5 I would execute

    ./new_day.sh 5

This creates the folder `day5/`, which will contain the script `day5.py`, a
file with a description of that day's puzzle (`day5.md`), and the input data
for that day, in a file named `input.txt`.

It's safe to run `./new_day.sh 5` again - the `day5.py` script is only
written if it doesn't exist yet.  The other files will be rewritten, but they 
are just static files, downloaded from the adventofcode.com website.

#### session_key.txt
For each day's puzzle, the problem is the same, but the input data you work
with (and the solution) is different for each participant.  So, you need to
be registered on the https://adventofcode.com site in order to participate,
so it can control who gets what input data.

On the website, your login is tracked using a cookie that contains a session
key.  For these scripts to access your personalized input data, you need to
dig out that session key and store it in the file `session_key.txt`, in the
same directory as this file.  That's easily done using the "Dev Tools" in
your browser - just look at the "Cookie" request header for any
adventofcode.com page you've loaded, after you've logged in.

A typical session key looks somthing like this,

    536167c4656455ffadded8c5751fc4b5616d4ad8722f5894ef76b9c403db4bf148c5c3f8571cf9aea038a55c7b3910a1bce63fc52666a92f881ab6aeea7c5101

#### dayN.py
This is my boilerplate code for adventofcode solutions.  I always solve a
challenge by first testing my code against the examples, given in the puzzle
description, and then running it against the input data.  The function
skeletons in `dayN.py` reflect this:

    solve() ........ code to solve the problem presented in part 1
    example1() ..... code to test solve() against the part-1 examples
    part1() ........ code to run solve() against the real part-1 input data

    solve2() ....... code to solve the problem presented in part 2
    example2() ..... code to test solve2() against the part-2 examples
    part2() ........ code to run solve2() against the real part-2 input data

The example* and part* functions check the answers against the expected answers.
The other functions are helper code to unpack a couple of the typical types of
input data you need to deal with.

There's no reason you need to structure your solution code this way, too.
If you use the `new_day.sh` machinery for your own solutions, you'll
probably want to modify `dayN.py` to suit your own coding style.


----
Tom Pollard :: November 28, 2023


