# General Commands and Instructions for connecting to PSC

Adapted from this link: [Reference](https://gist.github.com/clane9/2e843b708ab77f4f0526b3bf57268adb)

## 2. Start Interactive Session

Next, start an [interactive session](https://www.psc.edu/resources/bridges-2/user-guide-2-2/#interactive-sessions). I like to `ssh bridges2`, then create a [screen](https://linuxize.com/post/how-to-use-linux-screen/) session (in case my connection drops)

```sh
# Start or resume a screen session named "A"
screen -dR A
```

Then launch an interactive session

```sh
# Start an interactive session with 8 CPUs for 2 hours
interact -n 8 -t 2:00:00
```

```sh
# Start an interactive session with gpu node with one gpu for 2hrs
interact -p GPU-shared --gres=gpu:1 -t 2:00:00
```

Once the interactive session starts, make a note of what node you were assigned. Then you can just sleep inside the terminal (so the scheduler doesn't automatically cancel the job) and minimize the terminal--we won't need it anymore.