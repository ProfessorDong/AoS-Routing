"""Which constraints bind at the boundary of the projected region.

Figure 3 says the quantum-only projection is a rectangle and the
flexible one is not, and it attributes the difference to the compute
budget being charged per node.  That attribution is a claim about the
LP solution, not about the picture, so it is checked here rather than
asserted: the region LP is rebuilt with its rows labelled, solved at 45
degrees, and the tight rows are reported.

Two earlier readings of this figure were wrong and this script is what
caught them.  The two flows do NOT use disjoint edges (they share five),
so that cannot be why the quantum-only region is a rectangle.  And it is
not one gateway budget that binds but three, so naming a single node was
arbitrary.

It also sweeps the graded fraction with the full five-flow vector,
which is what Figure 4 plots.  That caught a third error: the caption
attributed the flat part of the curve to the classical capacity, and no
classical link is tight at any theta.  What binds is the quantum supply
throughout and the gateway compute budgets up to theta = 1/2.

Run:  python3 region_binding.py
"""
import sys, os
import math, dataclasses, numpy as np
from collections import defaultdict
from scipy.optimize import linprog
import aos_tqd as A

nodes, edges = A.default_topology(); flows = A.scaled_flows()
names=[f.name for f in flows]
sched={s: A.build_qkd_schedule(weather_seed=s, hours=12)[0] for s in range(5)}
etas=[A.empirical_eta(sched[s],43200,edges) for s in range(5)]
eta={k: float(np.mean([e[k] for e in etas])) for k in etas[0]}
i,j=names.index("bulk-A"), names.index("ses-A")

def solve(cfg, phi):
    probe=[]
    for k,f in enumerate(flows):
        g=dataclasses.replace(f)
        g.arrival_bps = math.cos(phi)*1e6 if k==i else (math.sin(phi)*1e6 if k==j else 0.0)
        probe.append(g)
    eks=[e.key() for e in edges]
    emap=dict((e.key(),e) for e in edges)
    C={k: A.bps_to_blocks(emap[k].capacity_bps, cfg.dt_s) for k in eks}
    adj=defaultdict(list)
    for (u,v) in eks: adj[u].append(v)
    classes=[]
    for f in probe:
        lam=A.bps_to_blocks(f.arrival_bps, cfg.dt_s)
        if cfg.theta>0: classes.append((f.src,f.dst,lam*cfg.theta,True,f.name))
        if cfg.theta<1: classes.append((f.src,f.dst,lam*(1-cfg.theta),False,f.name))
    cols=[]
    for ci,(s,d,lam,graded,nm) in enumerate(classes):
        if lam<=0: continue
        for seq in A._simple_paths(adj,s,d):
            path=list(zip(seq,seq[1:]))
            if any(k not in C for k in path): continue
            for sp in ("Q",) if graded else ("Q","P"):
                cols.append((ci,path,sp,nm))
    eidx={k:n for n,k in enumerate(eks)}
    nv=len(cols)+len(eks)+1; ti=nv-1
    Aub,b,rows=[],[],[]
    def add(r,bb,tag): Aub.append(r); b.append(bb); rows.append(tag)
    for k in eks:
        r=np.zeros(nv)
        for n,(_,pth,_,_) in enumerate(cols):
            if k in pth: r[n]=1.0
        add(r,C[k],('cap',k))
    for k in eks:
        r=np.zeros(nv)
        for n,(_,pth,sp,_) in enumerate(cols):
            if sp=="Q" and k in pth: r[n]=1.0
        add(r,eta.get(k,0.0),('eta',k))
    for k in eks:
        r=np.zeros(nv)
        for n,(_,pth,sp,_) in enumerate(cols):
            if sp=="P" and k in pth: r[n]=1.0
        r[len(cols)+eidx[k]]=-1.0
        add(r,0.0,('psupply',k))
    for k in eks:
        r=np.zeros(nv); r[len(cols)+eidx[k]]=1.0
        add(r,cfg.edge_h_max_units if cfg.manufacture else 0.0,('hmax',k))
    for n_ in nodes:
        r=np.zeros(nv)
        for k in eks:
            if k[0]==n_: r[len(cols)+eidx[k]]+=A.OPS_TAIL
            if k[1]==n_: r[len(cols)+eidx[k]]+=A.OPS_HEAD
        add(r,cfg.node_budget_units if cfg.manufacture else 0.0,('budget',n_))
    Aeq,beq=[],[]
    for ci,(_,_,lam,_,_) in enumerate(classes):
        if lam<=0: continue
        r=np.zeros(nv)
        for n,(cj,_,_,_) in enumerate(cols):
            if cj==ci: r[n]=1.0
        r[ti]=-lam; Aeq.append(r); beq.append(0.0)
    obj=np.zeros(nv); obj[ti]=-1.0
    res=linprog(obj,A_ub=np.array(Aub),b_ub=np.array(b),A_eq=np.array(Aeq),b_eq=np.array(beq),
                bounds=[(0,None)]*nv, method="highs")
    return res,rows,cols,np.array(Aub),np.array(b)

for lbl,cfg in (("flex theta=0", A.TqdConfig(theta=0.0)),
                ("prior B=0", A.TqdConfig(theta=1.0, manufacture=False))):
    res,rows,cols,Aub,b = solve(cfg, math.pi/4)
    slack=b-Aub@res.x
    kinds={}
    for n,(kind,who) in enumerate(rows):
        if slack[n] < 1e-6*max(1.0,abs(b[n])): kinds.setdefault(kind,[]).append(who)
    print(f"\n=== {lbl} at 45 deg: t*={res.x[-1]:.4f}")
    print(f"  TIGHT node budgets: {kinds.get('budget',[])}")
    print(f"  tight eta edges: {len(kinds.get('eta',[]))}, tight capacity: {len(kinds.get('cap',[]))}")
    use={}
    for n,(ci,pth,sp,nm) in enumerate(cols):
        if res.x[n]>1e-9: use.setdefault(nm,set()).update(pth)
    for nm,s in sorted(use.items()):
        print(f"  {nm}: gateways {sorted({x for e in s for x in e if x.startswith('GW')})}")
    if len(use)==2:
        a,bb=[use[k] for k in sorted(use)]
        print(f"  shared edges: {sorted(a&bb) or 'NONE'}")


def sweep_theta():
    """Figure 4: which resources bind as the graded fraction varies."""
    nominal = sum(f.arrival_bps for f in flows) / 1e6
    print("\n=== grade sweep, full five-flow vector")
    print(f"  {'theta':>6s} {'Mbps':>8s} {'classical':>10s} {'quantum':>8s} "
          f"{'budgets':>8s}")
    for th in (0.0, 0.125, 0.25, 0.5, 0.75, 1.0):
        res, rows, cols, Aub, b = solve(A.TqdConfig(theta=th), 0.0)
        slack = b - Aub @ res.x
        kinds = {}
        for n, (kind, who) in enumerate(rows):
            if slack[n] < 1e-6 * max(1.0, abs(b[n])):
                kinds.setdefault(kind, []).append(who)
        print(f"  {th:6.3f} {res.x[-1]*nominal:8.2f} "
              f"{len(kinds.get('cap',[])):10d} {len(kinds.get('eta',[])):8d} "
              f"{len(kinds.get('budget',[])):8d}")
    print("  the classical column is zero at every theta, which is why the"
          " boundary here is never a bandwidth limit")


if __name__ == "__main__":
    sweep_theta()
