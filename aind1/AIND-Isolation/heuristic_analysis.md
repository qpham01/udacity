# Isolation Heuristic Analysis

To build a good isolation game analysis, I start with information given in the lectures and
empirical results from running the tournament.py script.  Running the tournament.py script
unchanged showed that the AB Improved player does quite well against all comers with a winning
percentage north of 75%.  Also, in the "Solving 5x5 Isolation" lecture, we were told that the best
first move is always the center square for both agents.  Intuitively, biasing moves toward the
center of the board seems to provide the most options for later moves.  From this line of thought, 
one heuristic that I chose is a combination of center scoring and improved scoring evaluation 
function (custom_score3):

```
score = improved_score() - x * center_score()
```

The range of center\_score varies from 0.5 to 24.5, and the range of improved\_score varies 
from -7 to 7.  To limit the impact of center_score I set x to 0.2 to give the center\_score 
component a range from 0.1 through 4.9.  Since center_score is larger for player positions
farther from the center, the above evaluation penalizes moves further away from
the center of the board.

Here is the performance of this strategy in 100 matches against each non-random opponent:

**score = improved\_score() - x * center\_score() performance**
```
 Match #   Opponent     AB_Custom3
                        Won | Lost
    1       MM_Open     74  |  26
    2      MM_Center    86  |  14
    3     MM_Improved   66  |  34
    4       AB_Open     55  |  45
    5      AB_Center    54  |  46
    6     AB_Improved   52  |  48
```

The the above strategy did slightly better than AB Improved.  The next step is to add a library of
good opening moves.  While running the above test of 600 games, I modified the code to save the 
hashes of winning board positions from the first five moves of the games.  Given the determinism 
of minimax/alpha-beta, a winning game board position is like a full search of the game tree 
resulting in a win.  I also added the center of the board as a good opening move for both moving
first and second to form custom_score_2.

