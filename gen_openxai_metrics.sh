python3 EvaluateWithOpenXAI.py --model_name lr --data_name german > PLOTS/openxai_aucs.txt
python3 EvaluateWithOpenXAI.py --model_name ann --data_name german >> PLOTS/openxai_aucs.txt

python3 EvaluateWithOpenXAI.py --model_name lr --data_name heloc >> PLOTS/openxai_aucs.txt
python3 EvaluateWithOpenXAI.py --model_name ann --data_name heloc >> PLOTS/openxai_aucs.txt

python3 EvaluateWithOpenXAI.py --model_name lr --data_name compas >> PLOTS/openxai_aucs.txt
python3 EvaluateWithOpenXAI.py --model_name ann --data_name compas >> PLOTS/openxai_aucs.txt

python3 EvaluateWithOpenXAI.py --model_name lr --data_name adult >> PLOTS/openxai_aucs.txt
python3 EvaluateWithOpenXAI.py --model_name ann --data_name adult >> PLOTS/openxai_aucs.txt