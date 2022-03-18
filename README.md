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
```

## Usage

With default options
```bash
$ stack run
```

otherwise

```bash
$ stack exec -- edelwace-exe [options]
```

```
Usage: edelwace-exe [-l|--algorithm ALGORITHM] [-H|--host HOST] [-P|--port PORT]
                    [-i|--ace ID] [-p|--pdk PDK] [-v|--var VARIANT]
                    [-a|--act ACTIONS] [-o|--obs OBSERVATIONS] [-f|--path FILE]
  GACE RL Trainer

Available options:
  -l,--algorithm ALGORITHM DRL Algorithm, one of sac, td3, ppo (default: "sac")
  -H,--host HOST           Hym server host address (default: "localhost")
  -P,--port PORT           Hym server port (default: "7009")
  -i,--ace ID              ACE OP ID (default: "op2")
  -p,--pdk PDK             ACE Backend (default: "xh035")
  -v,--var VARIANT         GACE Environment Variant (default: "0")
  -a,--act ACTIONS         Dimensions of Action Space (default: 10)
  -o,--obs OBSERVATIONS    Dimensions of Observation Space (default: 42)
  -f,--path FILE           Checkpoint File Path (default: "./models")
  -h,--help                Show this help text
```

### Dependencies

- hasktorch
- libtorch-ffi
- mtl
- wreq
- aeson
- optparse-applicative

## Algorithms

Excessive use of Unicode and Strictness.

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
- [X] Implement TD3
- [ ] Implement PPO
- [X] Implement PER
- [ ] Visualization
- [ ] Wait for Normal Distribution in HaskTorch
- [ ] Remove strictness where unecessary
- [ ] Add agent loading ability
- [X] Command Line Options
