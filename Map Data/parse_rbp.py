import re,csv,json,collections,pathlib
p=pathlib.Path('/mnt/data/rbp_extract/USAMAP.RB3')
s=p.read_text(encoding='latin1')
sections={}
for m in re.finditer(r"(?m)^'\[([^]]+)\]\s*$",s):
 sections[m.group(1)]=s[m.end():]
for name in list(sections):
 sections[name]=sections[name].split("'[",1)[0]
def records(name):
 return [x.strip() for x in sections[name].splitlines() if x.strip() and not x.lstrip().startswith("'")]
regions=['NE','SE','NC','SC','PL','NW','SW']; dots={};nodes=[]
for i,region in enumerate(regions,1):
 for j,line in enumerate([v for v in records('re'+str(i)) if v.startswith('s,')]):
  if not line.startswith('s,'):continue
  _,x,y,code,*rest=line.split(','); nid=f'{region}_{j+1}'
  n={'id':nid,'region':region,'dot_number':j+1,'x':int(x),'y':int(y),'dot_style':code,'city':None};nodes.append(n);dots[(i,j+1)]=n
cities=[]
for line in records('city'):
 a=next(csv.reader([line]));
 if len(a)<5:continue
 region=a[1]; dot=int(a[4]); n=dots.get((regions.index(region)+1,dot))
 cities.append({'name':a[0],'region':region,'dot_number':dot,'node_id':n['id'] if n else None,'probability':a[2],'payout_number':a[3]})
 if n:n['city']=a[0]
rr=[]
for line in records('rr'):
 a=next(csv.reader([line]));
 if len(a)>=6:rr.append({'name':a[0],'short_name':a[2], 'price':int(a[5]),'color_code':a[3]})
# route section has 28 railroads; each railroad terminated by -1,-1; branches terminated by 0,0
lines=sections['rout'].splitlines(); data=[]; groups=[]; issues=[]
for line in lines:
 line=line.strip()
 if not line or line.startswith("'"):continue
 terminal='-1,-1' if line.endswith('-1,-1') else '0,0' if line.endswith('0,0') else None
 if not terminal:issues.append(('no_terminator',line));continue
 body=line[:-len(terminal)].rstrip(', ')
 path=[]
 for tok in body.split():
  a=[int(x) for x in tok.strip(',').split(',') if x]
  if not a:continue
  r=a[0]
  for d in a[1:]:
   loc=d%1000;node=dots.get((r,loc))
   if not node:issues.append((len(data),r,d))
   path.append({'node':node['id'] if node else None,'raw_dot':d,'region':r})
 groups.append(path)
 if terminal=='-1,-1':data.append(groups);groups=[]
if groups:data.append(groups)
print('RR',len(rr),'route RRs',len(data),'dots',len(nodes),'cities',len(cities),'missing refs',len(issues),'sample',issues[:30]);print('route lengths',[(rr[i]['short_name'] if i<len(rr) else i,len(routes)) for i,routes in enumerate(data)])
edges=[]
for i,paths in enumerate(data):
 for k,path in enumerate(paths):
  for a,b in zip(path,path[1:]):
   if a['node'] and b['node'] and a['node']!=b['node']:
    edges.append({'id':f'E{len(edges)+1:04d}','from':a['node'],'to':b['node'],'railroad':rr[i]['short_name'],'railroad_name':rr[i]['name'],'path_index':k+1,'from_raw_dot':a['raw_dot'],'to_raw_dot':b['raw_dot']})
for name,rows,cols in [('nodes.csv',nodes,['id','region','dot_number','x','y','dot_style','city']),('edges.csv',edges,['id','from','to','railroad','railroad_name','path_index','from_raw_dot','to_raw_dot']),('cities.csv',cities,['name','region','dot_number','node_id','probability','payout_number'])]:
 with open('/mnt/data/'+name,'w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)
with open('/mnt/data/rail_baron_graph.json','w') as f:json.dump({'source':'RBP USAMAP.RB3','note':'Route tokens normalized modulo 1000, pending validation of special encoded values.','nodes':nodes,'edges':edges,'railroads':rr,'cities':cities,'raw_route_paths':data,'unresolved_references':issues},f,indent=2)
print('edges',len(edges),'unique pairs',len(set((min(e['from'],e['to']),max(e['from'],e['to']),e['railroad']) for e in edges)))
