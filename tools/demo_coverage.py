"""Run the paper's three-goal example using the production allocation class."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from alfworld_decision_coverage import DecisionCoverageBuffer
b=DecisionCoverageBuffer()
rows=[dict(rollout=dict(policy_version='demo',gamefile='same-execution',rollout_id=0),goal=dict(task_type='task',object_type='object',feedback_steps=s,hsl_weight=1)) for s in ([0,1],[0,1],[0,2])]
w=b.goal_weights(rows,[])
print('Goal supports:',[r['goal']['feedback_steps'] for r in rows])
print('Initial probabilities:',[round(x/sum(w),6) for x in w])
w=b.goal_weights(rows,[rows[2]])
print('After reserving the third goal:',[round(x/sum(w),6) for x in w])
