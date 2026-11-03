"""Shared genomic-context markers and labels that avoid every plotted CpG."""
import numpy as np
import pandas as pd
from matplotlib.transforms import Bbox

STYLES={
 'Promoter':('#D55E00','^'), 'Enhancer':('#009E73','s'), 'Gene body':('#0072B2','o'),
 'Promoter + enhancer':('#A6761D','v'), 'Promoter + gene body':('#7B2CBF','D'),
 'Enhancer + gene body':('#008B8B','h'), 'Promoter + enhancer + gene body':('#CC79A7','P'),
 'Other context, opposite directions':('#333333','*'), 'Other context, same directions':('#B8BEC5','o')}


def scatter_context(ax,frame,annotation_path,x,y,out):
    ann=pd.read_csv(annotation_path,sep='\t').set_index('CpG_ID')
    records=[]
    for row in frame.itertuples():
        a=ann.loc[row.CpG_ID];regions=set(str(a.UCSC_RefGene_Group).split(';'))
        promoter=bool(regions & {'TSS200','TSS1500'})
        enhancer=str(a.Enhancer).strip().upper() in {'TRUE','1','1.0'}
        body='Body' in regions
        inverse=getattr(row,x)*getattr(row,y)<0
        kinds=[label for flag,label in [(promoter,'Promoter'),(enhancer,'enhancer'),(body,'gene body')] if flag]
        category=' + '.join(kinds).capitalize() if kinds else ('Other context, opposite directions' if inverse else 'Other context, same directions')
        # Preserve the initial capital of the standalone gene-body label.
        category=next(k for k in STYLES if k.lower()==category.lower())
        records.append(dict(CpG_ID=row.CpG_ID,promoter=promoter,enhancer=enhancer,gene_body=body,inverse_direction=inverse,display_category=category))
    display=pd.DataFrame(records).set_index('CpG_ID')
    for category,(color,marker) in STYLES.items():
        ids=display.index[display.display_category.eq(category)]
        part=frame.loc[frame.CpG_ID.isin(ids)]
        if part.empty:continue
        inverse=part.CpG_ID.map(display.inverse_direction).to_numpy(bool)
        legend_label=('unannotated' if category=='Other context, same directions' else
                      f'Unannotated, opposite direction (n={len(part)})' if category=='Other context, opposite directions' else
                      f'{category} (n={len(part)})')
        ax.scatter(part[x],part[y],s=85,color=color,marker=marker,
                   edgecolors=np.where(inverse,'#222222','white'),linewidths=np.where(inverse,1.25,.65),
                   zorder=3,label=legend_label)
    display.to_csv(out/'data/plotted_genomic_context.tsv',sep='\t')
    handles,labels=ax.get_legend_handles_labels()
    return handles,labels


def unobstructed_labels(fig,ax,frame,x,y,out):
    """Greedy pixel-space layout, checking text against all points and labels."""
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    coordinates=ax.transData.transform(frame[[x,y]].to_numpy(float))
    point_boxes=[Bbox.from_bounds(px-9,py-9,18,18) for px,py in coordinates]
    selected=frame.loc[frame[x].mul(frame[y]).lt(0)].copy()
    boxes=[];placements=[]
    # Position the widest labels first so narrow labels can fill smaller gaps.
    selected['text']=selected.CpG_ID+'\n'+selected.expression_genes_used.fillna('').str.replace(';',' / ',regex=False)
    selected=selected.sort_values('text',key=lambda col:col.str.len(),ascending=False)
    for row in selected.itertuples():
        target=ax.transData.transform((getattr(row,x),getattr(row,y)))
        probe=ax.text(0,0,row.text,fontsize=11,ha='center',va='center')
        extent=probe.get_window_extent(renderer);probe.remove()
        candidates=[]
        for fx in np.linspace(.07,.93,36):
            for fy in np.linspace(.07,.93,32):
                px,py=ax.transAxes.transform((fx,fy))
                box=Bbox.from_bounds(px-extent.width/2-5,py-extent.height/2-4,extent.width+10,extent.height+8)
                if not ax.bbox.contains(box.x0,box.y0) or not ax.bbox.contains(box.x1,box.y1):continue
                if any(box.overlaps(b) for b in point_boxes+boxes):continue
                candidates.append((np.linalg.norm(np.array([px,py])-target),px,py,box))
        if not candidates:raise RuntimeError(f'No unobstructed label position for {row.CpG_ID}')
        _,px,py,box=min(candidates,key=lambda item:item[0]);boxes.append(box)
        position=ax.transData.inverted().transform((px,py))
        ax.annotate(row.text,xy=(getattr(row,x),getattr(row,y)),xytext=position,
                    ha='center',va='center',fontsize=11,zorder=5,
                    arrowprops=dict(arrowstyle='->',color='#666666',lw=.8,shrinkA=5,shrinkB=8),
                    bbox=dict(facecolor='white',edgecolor='none',pad=.3))
        placements.append(dict(CpG_ID=row.CpG_ID,label_x=position[0],label_y=position[1]))
    assert all(not box.overlaps(point) for box in boxes for point in point_boxes)
    pd.DataFrame(placements).to_csv(out/'data/label_positions.tsv',sep='\t',index=False)
    selected.drop(columns='text').to_csv(out/'data/displayed_CpG_gene_labels.tsv',sep='\t',index=False)
