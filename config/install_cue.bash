#!/bin/bash

wget -O cue.tar.gz https://github.com/cue-lang/cue/releases/download/v0.4.2/cue_v0.4.2_linux_amd64.tar.gz
tar -zxvf cue.tar.gz -C /tmp
cp /tmp/cue ~/.local/bin/
rm -rf cue.tar.gz