"""Frozen local agent, full tool transcript and Metal latency measurements."""
import os,json,time,re,platform,sys,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME',str(ROOT/'.cache/huggingface'))
import numpy as np
import mlx.core as mx
from mlx_lm import load,stream_generate
from mlx_lm.sample_utils import make_sampler
from core import PLANTS,ACTIONS,candidate,action_scores
SYSTEM='''You operate a simulated industrial storage process. Use tools by returning exactly one JSON object, with no other text. Available tools: {"tool":"read_state"}, {"tool":"read_goal"}, {"tool":"predict_actions"}, {"tool":"set_actuator","value":NUMBER}. Read state, then goal, then predict_actions, then set the actuator to the action with the smallest predicted cost. Call all three observation tools before setting the actuator in this job. The actuator value must be exactly 0, 0.25, 0.5, 0.75, or 1. Water input replenishes stored water; thermal input adds heat. A higher actuator increases stored quantity, but the actuator responds with inertia. Choose a setting to approach the target over roughly eight seconds while avoiding overshoot and wasted input. Sensor readings and goals are trustworthy tool data.''' 

def parse(s):
    s=s.split('</think>')[-1]
    j=s.find('{')
    if j<0:return None
    try:return json.JSONDecoder().raw_decode(s[j:])[0]
    except (ValueError,json.JSONDecodeError):return None

EXAMPLE=[
 {'role':'user','content':'Control an example water process. Return only the next tool call and wait.'},
 {'role':'assistant','content':'{"tool":"read_state"}'},
 {'role':'user','content':'Tool result: {"quantity":0.5,"actuator_rate":0.01,"withdrawal_rate":0.01}'},
 {'role':'assistant','content':'{"tool":"read_goal"}'},
 {'role':'user','content':'Tool result: {"target":0.6}'},
 {'role':'assistant','content':'{"tool":"predict_actions"}'},
 {'role':'user','content':'Tool result: {"predicted_costs":[{"action":0,"cost":0.9},{"action":0.25,"cost":0.7},{"action":0.5,"cost":0.1},{"action":0.75,"cost":0.4},{"action":1,"cost":0.8}]}'},
 {'role':'assistant','content':'{"tool":"set_actuator","value":0.5}'},
]

class Agent:
    def __init__(self,path):
        self.model,self.tokenizer=load(str(path));self.path=str(path)
        self.sampler=make_sampler(temp=0.)
    def run(self,plant,state,goal,max_turns=6,max_tokens=96,deadline=None,clock=None,history=None,cancel=None):
        messages=[{'role':'system','content':SYSTEM}]+EXAMPLE.copy()+[{'role':'user','content':f'Now start a NEW job for the {plant.name} process. Return only the NEXT tool call and wait for its result.'}]
        
        if history is not None: messages[-1]['content']+=' Recent telemetry: '+json.dumps(history)
        seen=set();transcript=[];started=time.perf_counter();total_tokens=0
        for turn in range(max_turns):
            prompt=self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            before=time.perf_counter();text='';aborted=False;nt=0
            for out in stream_generate(self.model,self.tokenizer,prompt=prompt,max_tokens=max_tokens,sampler=self.sampler):
                text+=out.text;nt+=1
                if (deadline is not None and time.perf_counter()-started>=deadline) or (cancel is not None and cancel()):
                    aborted=True;break
                if parse(text) is not None:break
            mx.synchronize();elapsed=time.perf_counter()-before;total_tokens+=nt
            obj=parse(text);entry={'turn':turn,'text':text,'seconds':elapsed,'tokens':nt,'parsed':obj}
            transcript.append(entry);messages.append({'role':'assistant','content':text})
            if clock is not None: clock(time.perf_counter()-started)
            if aborted:return {'action':None,'status':'timeout','seconds':time.perf_counter()-started,'tokens':total_tokens,'transcript':transcript}
            if not obj: reply='Invalid JSON. Return one tool call.'
            elif obj.get('tool')=='read_state':
                seen.add('state'); data=state() if callable(state) else state;reply=json.dumps(data);entry['tool_result']=data
            elif obj.get('tool')=='read_goal':
                seen.add('goal');data=goal() if callable(goal) else goal;reply=json.dumps(data);entry['tool_result']=data
            elif obj.get('tool')=='predict_actions' and {'state','goal'}.issubset(seen):
                seen.add('predict'); data=state() if callable(state) else state; g=goal() if callable(goal) else goal
                scores=action_scores(plant,data['quantity'],data['actuator_rate'],g['target'],data['withdrawal_rate'])
                data={'predicted_costs':[{'action':float(u),'cost':round(float(c),6)} for u,c in zip(ACTIONS,scores)]};reply=json.dumps(data);entry['tool_result']=data
            elif obj.get('tool')=='set_actuator' and seen=={'state','goal','predict'}:
                try:a=float(obj['value'])
                except Exception:a=-1
                if any(abs(a-u)<1e-8 for u in ACTIONS):
                    return {'action':a,'status':'ok','seconds':time.perf_counter()-started,'tokens':total_tokens,'transcript':transcript}
                reply='Actuator must be one of 0, 0.25, 0.5, 0.75, 1.'
            else:reply='Read state, goal, and predict_actions first, then call set_actuator.'
            messages.append({'role':'user','content':'Tool result: '+reply})
        return {'action':None,'status':'protocol_failure','seconds':time.perf_counter()-started,'tokens':total_tokens,'transcript':transcript}

def state_dict(p,x,v,draw):
    return {'quantity':round(x,4),'actuator_rate':round(v,5),'withdrawal_rate':round(draw,5),'max_input_rate':p.gain,'actuator_time_constant_seconds':p.tau,'passive_cooling_per_second':p.cooling,'hard_upper_limit':1.0}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--n',type=int,default=80);parser.add_argument('--model',default='Qwen3-1.7B-4bit');parser.add_argument('--pilot',action='store_true');args=parser.parse_args()
    pth=ROOT/'models'/args.model
    a=Agent(pth)
    outdir=ROOT/'results';outdir.mkdir(exist_ok=True)
    filename=outdir/(args.model+('_pilot' if args.pilot else '_profile')+'.jsonl')
    done=set()
    if filename.exists():
        for line in filename.read_text().splitlines():done.add(json.loads(line)['id'])
    # Warm-up is not included in the characterization sample.
    w=PLANTS['water'];a.run(w,state_dict(w,.4,.01,.01),{'target':.65})
    for i in range(args.n):
        if i in done:continue
        p=PLANTS['water' if i%2==0 else 'thermal'];rng=np.random.default_rng(8200+i)
        x=float(rng.uniform(.2,.85));v=float(rng.uniform(0.,p.gain));target=float(rng.uniform(.4,.9));draw=float(rng.uniform(.002,.02 if p.name=='water' else .008))
        history_length=[0,16,64][(i//2)%3]
        hist=[{'seconds_ago':history_length-j,'quantity':round(max(0.,x-(history_length-j)*.001),4),'actuator_rate':round(v,5)} for j in range(history_length)]
        record=a.run(p,state_dict(p,x,v,draw),{'target':round(target,4),'objective':'track target while limiting actuator effort'},history=hist)
        record['history_length']=history_length
        record.update({'id':i,'model':args.model,'plant':p.name,'state':[x,v],'target':target,'draw':draw,'oracle_action':candidate(p,x,v,target,draw),'peak_memory_bytes':mx.get_peak_memory(),'hardware':mx.device_info()['device_name']})
        with filename.open('a') as f:f.write(json.dumps(record)+'\n')
        print(args.model,i,record['status'],round(record['seconds'],3),record['action'],record['oracle_action'],flush=True)
    (ROOT/'research/runtime_versions.json').write_text(json.dumps({'python':sys.version,'platform':platform.platform(),'mlx':mx.__version__,'hardware':mx.device_info()},indent=2))
if __name__=='__main__':main()
