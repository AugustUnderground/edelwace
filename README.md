# EDELWAC²E

Reinforcement Learning Algorithms for
[GAC²E](https://github.com/AugustUnderground/gace) via
[Hym](https://github.com/AugustUnderground/hym) in Haskell with
[HaskTorch](https://github.com/hasktorch/hasktorch).

## Setup

LibTorch is required, as per HaskTorch Tutorial, and must be symlinked into
this directory. Then source `setenv` in your shell.

For training [Hym](https://github.com/AugustUnderground/hym) must be up
and running.

```bash
$ source setenv
$ stack build
$ stack run
```

### Dependencies

- hasktorch
- libtorch-ffi
- mtl
- wreq
- aeson

## Algorithms

Excessive use of Unicode.

### SAC

...

### TD3

...

### PPO

...

## Results

...

## TODO

- [X] Implement SAC
- [ ] Implement TD3
- [ ] Implement PPO
- [X] Implement PER
- [ ] Visualization
- [ ] Wait for Normal Distribution in HaskTorch
