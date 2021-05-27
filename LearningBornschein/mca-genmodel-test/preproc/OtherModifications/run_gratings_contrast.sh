filename="output.txt"
# nv_list='0.0 0.25 0.5 0.75 1.0'
#nv_list='0.0'
contrasts='0.0 0.1 0.15 0.2 0.25 0.5 0.75 1.0 1.5 2.0 2.5'
# constrasts='0.0'
g_ph='1 2 3 4 5'
# g_ph='1'


# for con in $contrasts
# do
# 	# to_print="Processing con ${con}"
# 	# echo '============================================================' 2>> $to_print
# 	python preproc-zca-modified-contrast.py patches-8.h5 $con
# done



for ph in $g_ph
do
  for con in $contrasts
  do
    to_print="Processing contrast ${con} phase ${ph}"
    # filename1 = "Gratings_latest${ph}_${constrast}.npy"
    # filename2 = "NatImZCA_${nv}.npy"
    #echo '============================================================' 2>> $to_print
    # python3 teacher_student.py --input_dim 500 --student_h_size 6 --teacher_path $teacher_path  --nonlinearity sigmoid --pruning_choice dpp_edge  --mode normal  --trained_weights student_6.pth --procedure pruning --num_masks 50 --k $var >> $filename
    python preproc-Grating-modified-contrast.py Gratings_latest${ph}_1.0.npy NatImZCA-mf_con_1.0.npy ${con}
  done
done

