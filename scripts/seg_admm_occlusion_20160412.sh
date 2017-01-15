#Do the segmentation (without the group LASSO penalty these can be run independently)
./seg_admm_occlusion.py --rframes 1,501 --iframes 1,251,501,751,1001,1251 -n 3000 ./simmatrix/20160412_occ/
./seg_admm_occlusion.py --rframes 1,501 --iframes 1501,1751,2001,2251,2501,2751 -n 3000 ./simmatrix/20160412_occ/
./seg_admm_occlusion.py --rframes 1,501 --iframes 3001,3251,3501,3751,4001,4251 -n 3000 ./simmatrix/20160412_occ/
./seg_admm_occlusion.py --rframes 1,501 --iframes 4501,4751 -n 3000 ./simmatrix/20160412/

#Then run the path continuation
./continue_mfsf_occlusion.py --rframes 1,501 --iframes 1,251,501,751,1001,1251,1501,1751,2001,2251,2501,2751,3001,3251,3501,3751,4001,4251,4501,4751 ./simmatrix/20160412_occ/ ./mfsf_output/

#Then make some visualizations 


#(mesh vis)