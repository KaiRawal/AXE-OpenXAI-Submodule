python3 EvaluateWithAXE.py --model_name lr --data_name german > PLOTS/axe_aucs.txt
python3 EvaluateWithAXE.py --model_name ann --data_name german >> PLOTS/axe_aucs.txt

python3 EvaluateWithAXE.py --model_name lr --data_name heloc >> PLOTS/axe_aucs.txt
python3 EvaluateWithAXE.py --model_name ann --data_name heloc >> PLOTS/axe_aucs.txt

python3 EvaluateWithAXE.py --model_name lr --data_name compas >> PLOTS/axe_aucs.txt
python3 EvaluateWithAXE.py --model_name ann --data_name compas >> PLOTS/axe_aucs.txt

python3 EvaluateWithAXE.py --model_name lr --data_name adult >> PLOTS/axe_aucs.txt
python3 EvaluateWithAXE.py --model_name ann --data_name adult >> PLOTS/axe_aucs.txt