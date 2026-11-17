import unittest
from alfworld_verified_recovery import choose_jobs as baseline
from alfworld_continuation_compression import choose_jobs

class ContinuationTests(unittest.TestCase):
    def test_original_proposals_and_seeds_preserved(self):
        rows=[dict(won=False,rollout_id=i,task_description='put a book',trajectory=[dict(action='look',result_observation='desk')]*30) for i in range(6)]
        bank=[dict(id='a',goal='put a book',context='desk',text='example')]
        old=baseline(rows,bank,5,42);new=choose_jobs(rows,bank,5,42)
        self.assertEqual(old,[{k:v for k,v in j.items() if k!='max_steps'} for j in new])
        self.assertTrue(all(j['max_steps']==24 for j in new));self.assertEqual(len(new),8)

if __name__=='__main__':unittest.main()
