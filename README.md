QPCA + KDD

# Tutorial for ARC UTSA Supercomputer
#### Commands to enter in ARC UTSA
ssh ats852@arc.utsa.edu

srun -p compute1 -n 1 -t 00:15:00 --pty bash
srun -p compute1 -n 1 -t 00:05:00 --cpus-per-task=32 --pty bash

module restore experiments

conda activate QUANTUM

#run the code

cd QPCA-KDD/

python -m qgan.train_ibm --arc

Journal:
CPU_EPOCHS=50 QPU_EPOCHS=50 python -m qgan.train_journal --mode full --conditions all
CPU_EPOCHS=50 QPU_EPOCHS=50 python -m qgan.train_journal --mode full --conditions all
CPU_EPOCHS=50 QPU_EPOCHS=50 python -m qgan.train_journal --mode full --conditions all

CPU_EPOCHS=50 QPU_EPOCHS=0 python -m qgan.train_journal --mode full --conditions cpu
CPU_EPOCHS=0 QPU_EPOCHS=50 python -m qgan.train_journal --mode full --conditions qpu


#### Vim Document
vim name_file.py            oppen file

a                           write file

ESC + dd                    delete the entire line

ESC + :w                    save the writing on file

ESC + :q                    close file

ESC + :wq                   save and close file

wget -c https://osf.io/ct6fd/download -O sleep_edf_dataset.zip

wget -c https://osf.io/z5xea/download -O sleep2.zip

wget -c https://osf.io/e3mf4/download -O sleep3.zip

wget -c https://osf.io/v3asr/download -O sleep4.zip

wget -c https://osf.io/h3p8b/download -O sleep5.zip

wget -c https://osf.io/mh76w/download -O sleep6.zip

wget -c https://osf.io/cqexb/download -O sleep7.zip

wget -c https://osf.io/e3ukr/download -O sleep8.zip

wget -c https://osf.io/h3qvb/download -O sleep9.zip

wget -c https://osf.io/2rsvf/download -O sleep10.zip

unzip -t sleep3.zip
unzip -t sleep4.zip
unzip -t sleep5.zip
unzip -t sleep6.zip
unzip -t sleep7.zip
unzip -t sleep8.zip
unzip -t sleep9.zip
unzip -t sleep10.zip


for f in sleep3.zip sleep4.zip sleep5.zip sleep6.zip sleep7.zip sleep8.zip sleep9.zip sleep10.zip; do
    unzip -n "$f" -d sleep_edf/
done