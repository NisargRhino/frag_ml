1. git clone the repository
2. Install all the neccesary packages.
3. Run train_model.py first in order to train the machine learning algorithm. Make sure to input the correct paths of the CSV.
4. Run frag_predict.py in order to predict the best fragment for the inputted drug. Make sure to put the correct paths of the CSV.
5. combine_frag.py combines 2 fragments into one drug. 
6. gen_data.py generates a dataset of drug SMILES and its frag SMILES. Enter a csv of drug SMILES, protein, and coordinates.
   Example input through terminal : python gen_data.py --input_csv "input.csv" --mol2_path "mTOR.mol2" --docking_dir "dockdir" --target_name "mTOR" --center_coords 68.0658 -5.1678 -54.97 --box_sizes 98.194 95.5592 116.24 --output_path "output.csv"
   
