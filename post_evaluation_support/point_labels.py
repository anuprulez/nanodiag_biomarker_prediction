"""Direct CpG/gene labels with deterministic collision avoidance and no arrows."""
import textwrap
import numpy as np
from matplotlib.transforms import Bbox


def label_cpg_points(ax,frame,x,y,genes,fontsize=8,inverse_only=False):
    mask=frame[x].notna()&frame[y].notna()&~(frame[x].lt(0)&frame[y].lt(0))
    if inverse_only:
        mask &= (frame[x].lt(0)&frame[y].gt(0)) | (frame[x].gt(0)&frame[y].lt(0))
    selected=frame.loc[mask]
    texts=[]
    for row in selected.to_dict('records'):
        label=row['CpG_ID']+'\n'+textwrap.fill(' / '.join(str(row[genes]).split(';')),width=23,break_long_words=False)
        texts.append(ax.text(row[x],row[y],label,fontsize=fontsize,ha='center',va='center',zorder=5))
    ax.figure.canvas.draw();renderer=ax.figure.canvas.get_renderer()
    points=ax.transData.transform(frame[[x,y]].to_numpy())
    targets=ax.transData.transform(selected[[x,y]].to_numpy())
    dimensions=[(t.get_window_extent(renderer).width,t.get_window_extent(renderer).height) for t in texts]
    area=ax.get_window_extent();zero=ax.transData.transform((0,0));placed=[]
    angles=np.deg2rad(np.array([45,-45,135,-135,0,90,180,-90]+list(range(0,360,15))))
    # Allocate room for the most crowded labels first.
    density=[np.sum(np.linalg.norm(targets-p,axis=1)<90) for p in targets]
    order=sorted(range(len(texts)),key=lambda i:(-density[i],i))
    for index in order:
        target=targets[index];width,height=dimensions[index]
        px=selected.iloc[index][x];py=selected.iloc[index][y]
        xmin=zero[0]+4 if px>=0 else area.x0+4
        xmax=area.x1-4 if px>=0 else zero[0]-4
        ymin=zero[1]+4 if py>=0 else area.y0+4
        ymax=area.y1-4 if py>=0 else zero[1]-4
        best=None;best_score=float('inf');found=False
        for radius in np.arange(10,max(area.width,area.height)+20,8):
            for angle in angles:
                center=target+radius*np.array([np.cos(angle),np.sin(angle)])
                box=Bbox.from_bounds(center[0]-width/2-2,center[1]-height/2-2,width+4,height+4)
                if box.x0<xmin or box.x1>xmax or box.y0<ymin or box.y1>ymax:continue
                overlaps=sum(box.overlaps(other) for other in placed)
                point_overlap=np.any((points[:,0]>=box.x0-2)&(points[:,0]<=box.x1+2)&(points[:,1]>=box.y0-2)&(points[:,1]<=box.y1+2))
                score=overlaps*1e8+int(point_overlap)*1e6+radius*radius
                if score<best_score:best=(center,box);best_score=score
                if overlaps==0 and not point_overlap:found=True;break
            if found:break
        if best is None:raise RuntimeError('Not enough plotting space for a label; enlarge the figure')
        center,box=best;texts[index].set_position(ax.transData.inverted().transform(center));placed.append(box)
    return selected[['CpG_ID',genes,x,y]].copy()
