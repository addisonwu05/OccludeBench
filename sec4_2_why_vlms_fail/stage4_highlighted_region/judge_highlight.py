import glob, json, os, sys
from dotenv import load_dotenv
from openai import OpenAI
import anthropic
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.abspath(os.path.join(HERE,'..','..'))
load_dotenv(os.path.join(ROOT,'.env'))
CL={'openai':OpenAI(api_key=os.getenv('OpenAI_API_KEY')),'anthropic':anthropic.Anthropic(api_key=os.getenv('Anthropic_API_KEY'))}
JUDGES=['gpt-5.4','claude-opus-4-6']
P=("You are evaluating whether a vision model correctly identified an object.\n\n"
   "The model was shown an occluded image with a reflection/shadow region highlighted, "
   "and asked to name the underlying object.\n\nGround truth: \"{gt}\"\n\nModel response: \"{resp}\"\n\n"
   "Does the response correctly identify the same underlying OBJECT as the ground truth? "
   "The ground truth may be phrased as 'reflection of a car' or 'car shadow'; the response only needs "
   "to identify the underlying object (e.g. 'car') and need NOT state it is a shadow/reflection. "
   "Allow reasonable variation in phrasing/synonyms. Answer only 'yes' or 'no'.")
def jc(gt,resp,jm):
    pr=P.format(gt=(gt or '').strip(),resp=(resp or '').strip())
    if jm.startswith('claude'):
        v=CL['anthropic'].messages.create(model=jm,temperature=0,max_tokens=10,messages=[{'role':'user','content':pr}]).content[0].text
    else:
        v=CL['openai'].chat.completions.create(model=jm,temperature=0,messages=[{'role':'user','content':pr}]).choices[0].message.content
    return v.strip().lower().startswith('yes')
for model in sys.argv[1:]:
    d=json.load(open(f'identify_highlight_{model}_results.json'))
    out=f'identify_highlight_{model}_judged_results.json'
    judged=json.load(open(out)) if os.path.exists(out) else []
    by={r['id']:r for r in judged}
    for r in d:
        rec=by.get(r['id'])
        if rec is None:
            rec=dict(r); rec['correct']={}; judged.append(rec); by[r['id']]=rec
        rec.setdefault('correct',{})
        for jm in JUDGES:
            if jm not in rec['correct']: rec['correct'][jm]=jc(r['ground_truth'],r['response'],jm)
        rec['consensus_correct']=any(rec['correct'].values())
        json.dump(judged,open(out,'w'),indent=2)
    c=sum(1 for r in judged if r.get('consensus_correct'))
    print(f"{model}: {c}/{len(judged)} = {c/len(judged):.1%}")
