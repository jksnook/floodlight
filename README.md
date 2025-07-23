# Floodlight
An NNUE Chess Engine derived from my previous piece-square table engine Spotlight: https://github.com/jksnook/spotlight

## Features:

### Move generation
* Fully legal move generation with magic bitboards
* around 45 million nps in perft on my (mediocre) system without bulk counting

### Search:
* Transposition Table
* Iterative Deepening
* Principal variation search
* Aspiration Windows
* Null move pruning
* Futility pruning
* Reverse futility pruning
* Internal iterative reductions
* Late move reductions
* Late move pruning
* SEE pruning
* Lazy SMP

### Move ordering
* TTmove
* captures ordered by SEE
* Killer moves
* Butterfly history heuristic

### Evaluation
* A simple (768 -> 128)x2 -> 1 NNUE architecture with AVX2 SIMD support
* Trained using bullet: https://github.com/jw1912/bullet
* Trained entirely with self-generated data (This is currently the limiting factor for network size as datagen takes a very long time)

## Planned Improvements:
* Refactor movegen and move ordering
* Bigger NNUE architecture