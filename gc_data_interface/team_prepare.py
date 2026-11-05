"""Prepare the identical numeric training/validation cache for all teammates."""
import argparse,json
from team_data import TeamData


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True);p.add_argument('--cache-dir',required=True)
    args=p.parse_args();data=TeamData(args.root,args.cache_dir)
    train=data.raw('train');val=data.raw('val')
    print(json.dumps({'data_fingerprint':data.fingerprint,'train_samples':len(train['obs']),
                      'val_samples':len(val['obs']),'test_read':False,'protocol':data.protocol},indent=2))


if __name__=='__main__':main()
