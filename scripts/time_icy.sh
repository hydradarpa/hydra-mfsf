#!/bin/bash
for MF in {4..50..5}
do
	./icy_particletracker.py ../hydra/video/20160412a/stk_0001/frames8/ tracks/20160412/stk_0001.pkl -maxfiles $MF
done