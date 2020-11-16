root_dir='./'
curr_dir='./'

input_fname='bold_dico_bold7Tp1_to_subjbold7Tp1.nii.gz'

for i in `seq 7 9`; do
    sub_id=$(printf "%03d" $i)
    for j in `seq 1 8`; do
        run_id=$(printf "%03d" $j)        
        input_dir=$root_dir'/sub'$sub_id'/BOLD/task002_run'$run_id/$input_fname
        ref_dir=$root_dir'/sub'$sub_id'/templates/bold7Tp1/brain.nii.gz'
        warp_dir=$root_dir'/sub'$sub_id'/templates/bold7Tp1/in_grpbold7Tp1/subj2tmpl_warp.nii.gz'
        output_dir=$curr_dir'/sub'$sub_id'_run'$run_id'.nii.gz'
        npz_output=$curr_dir'/sub'$sub_id'_run'$run_id'.npz'
        if [[ ( ! -f "$npz_output" )  && ( -f "$input_dir" ) ]]; then
            echo "process sub$sub_id run $run_id"
            applywarp -i $input_dir -r $ref_dir -w $warp_dir -o $output_dir
            python resize.py $output_dir
            rm $output_dir
        fi
    done
done





