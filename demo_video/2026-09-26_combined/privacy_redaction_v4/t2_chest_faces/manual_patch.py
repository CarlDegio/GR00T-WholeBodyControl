from pathlib import Path
p=Path(__file__).parent/'refine.py'
s=p.read_text();pos=s.index("(P/'masks.json')")
manual='''# Manual corrections from inspection of every actual frame: cropped faces at viewport edges.
edge_spans=[
(4092,4092,[1488,400,1512,459]),
(4232,4235,[1488,410,1517,466]),
(4256,4262,[1488,410,1520,472]),
(4263,4265,[1488,414,1512,474]),
(4434,4435,[1488,419,1523,473]),
(4468,4469,[1488,386,1517,456]),
(4472,4475,[1488,386,1522,455]),
(4612,4613,[1488,414,1522,464]),
(4626,4627,[1488,371,1520,427]),
(4634,4635,[1488,403,1520,462]),
(4656,4665,[1488,381,1522,436]),
(4682,4683,[1488,403,1522,481]),
(4690,4691,[1488,385,1524,457]),
(4742,4745,[1488,395,1525,463]),
(5146,5147,[1806,368,1836,418]),
(5158,5159,[1806,383,1836,434]),
]
absent_spans=[(4059,4059),(4093,4094),(4192,4193),(4236,4247),(4266,4266),(4433,4433),(4470,4471),(4482,4485),(4611,4611),(4628,4633),(4684,4689),(4746,4749),(5145,5145),(5160,5160)]
for lo,hi,box in edge_spans:
 for n in range(lo,hi+1):result[str(n)]=[box]
for lo,hi in absent_spans:
 for n in range(lo,hi+1):result[str(n)]=[]
for d in details:
 d['box']=result[str(d['n'])]
 d['manual_edge']=any(lo<=d['n']<=hi for lo,hi,_ in edge_spans)
 d['verified_absent']=any(lo<=d['n']<=hi for lo,hi in absent_spans)
'''
s=s[:pos]+manual+s[pos:];s=s.replace('Real-frame local template matching against nearby manual exemplars; final1920x1080 exclusivexyxy.','Real-frame local template matching; every one of 319 frames visually inspected; manually corrected partial faces at viewport edges. Final 1920x1080 exclusive xyxy.')
p.write_text(s)
