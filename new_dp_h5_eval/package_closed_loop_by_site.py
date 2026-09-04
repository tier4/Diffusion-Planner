"""Collect the complete route rosbags referenced by closed-loop by-site list."""
import argparse, json, shutil
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('site_list',type=Path); ap.add_argument('out',type=Path); ap.add_argument('--ros-root',type=Path,default=Path('/mnt/storage_rdma/diffusion_planner/rosbags_from_label')); a=ap.parse_args()
    out=a.out.resolve(); out.mkdir(parents=True,exist_ok=False)
    data=json.loads(a.site_list.read_text()); manifest=[]
    for site, vals in data.items():
      for text in vals:
        q=Path(text); i=q.parts.index('20260817_closed_loop_dataset'); rel=Path(*q.parts[i+1:]); base=a.ros_root/rel.parent.parent.parent
        found=None
        for split in ('valid','train','auto'):
          c=base/split/rel.parts[-2]/rel.parts[-1]
          if (c/'log_file_info.json').is_file(): found=(split,c); break
        if found is None: raise FileNotFoundError(f'no rosbag for {text}')
        split,c=found; stage=out/'rosbags'/rel.parts[0]/rel.parts[1]/split/rel.parts[-2]/rel.parts[-1]
        stage.parent.mkdir(parents=True,exist_ok=True); shutil.copytree(c,stage,symlinks=True)
        m=c.parents[2]/'map'
        if m.is_dir(): shutil.copytree(m,out/'rosbags'/rel.parts[0]/rel.parts[1]/'map',symlinks=True,dirs_exist_ok=True)
        manifest.append({'site':site,'source_dataset_path':text,'source_rosbag':c.as_posix(),'split':split,'package_rosbag':stage.relative_to(out).as_posix()})
    shutil.copy2(a.site_list,out/'path_list_closed_loop_by_site.json')
    (out/'manifest.json').write_text(json.dumps({'routes':manifest},indent=2)+'\n')
    print(f'packaged {len(manifest)} complete routes')
if __name__=='__main__': main()
