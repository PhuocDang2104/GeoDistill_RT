"""CPU contracts; not a substitute for real CUDA teacher/student qualification."""
import copy
import json
import random
import tempfile
import unittest
import zipfile
from pathlib import Path
import cv2
import numpy as np
import torch
from utils import identity,read_json,sha256
from full_data import SIZE,ResumableBatches,FullKITTI,read_sample,make_index
from teacher_cache import connection,commit_record,already_done,cache_path,validate_arrays,audit,predict_row
from server_runtime.recovery import capture_rng,restore_rng,commit_checkpoint,load_checkpoint,alias_checkpoint


class ProductionContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.cfg={'dataset_root':str(self.root/'kitti'),'work':str(self.root/'work'),
                  'teacher_root':str(self.root/'teachers'),'expected_counts':{'train':2,'val':2,'test':1}}

    def tearDown(self):self.temp.cleanup()

    def fixtures(self):
        root=Path(self.cfg['dataset_root'])
        for split,drive in (('train','2011_09_26_drive_0001_sync'),('val','2011_09_26_drive_0002_sync')):
            for frame in (0,1):
                name=f'{frame:010d}.png'
                paths=[root/split/drive/'proj_depth'/'groundtruth'/'image_02'/name,
                       root/split/drive/'proj_depth'/'velodyne_raw'/'image_02'/name,
                       root/'raw'/'2011_09_26'/drive/'image_02'/'data'/name]
                for p in paths:p.parent.mkdir(parents=True,exist_ok=True)
                gt=np.zeros(SIZE,np.uint16);gt[200:300,300:600]=2560
                sparse=np.zeros_like(gt);sparse[250,400]=2560
                cv2.imwrite(str(paths[0]),gt);cv2.imwrite(str(paths[1]),sparse)
                cv2.imwrite(str(paths[2]),np.zeros((*SIZE,3),np.uint8))
            c=root/'raw'/'2011_09_26'/'calib_cam_to_cam.txt'
            c.write_text('P_rect_02: 700 0 600 0 0 700 170 0 0 0 1 0\n')
        test=root/'depth_selection'/'test_depth_completion_anonymous'
        for folder in ('image','velodyne_raw','intrinsics'):(test/folder).mkdir(parents=True)
        cv2.imwrite(str(test/'image'/'0000000000.png'),np.zeros((*SIZE,3),np.uint8))
        cv2.imwrite(str(test/'velodyne_raw'/'0000000000.png'),np.zeros(SIZE,np.uint16))
        (test/'intrinsics'/'0000000000.txt').write_text('700 0 600 0 700 170 0 0 1')
        return make_index(self.cfg)

    def test_index_decode_and_val_no_teacher(self):
        report=self.fixtures()
        self.assertEqual(report['counts'],{'train':2,'val':2,'test':1})
        ds=FullKITTI(self.cfg,'train');sample=ds[0]
        self.assertEqual(float(sample['gt'].max()),10.)
        self.assertEqual(tuple(sample['K'].shape),(3,3))
        self.assertNotIn('teacher',FullKITTI(self.cfg,'val')[0])
        with self.assertRaises(ValueError):FullKITTI(self.cfg,'val',True)
        # Mutating source inputs cannot reuse the same frozen index.
        row=ds.rows[0];p=Path(self.cfg['dataset_root'])/row['rgb']['path']
        with p.open('ab') as f:f.write(b'changed')
        with self.assertRaises(RuntimeError):ds[0]

    def test_sampler_final_batch_and_consumed_resume(self):
        full=list(ResumableBatches(11,4,42,epoch=2))
        self.assertEqual(sorted(sum(full,[])),list(range(11)))
        self.assertEqual(len(full[-1]),3)
        self.assertEqual(list(ResumableBatches(11,4,42,epoch=2,start=1)),full[1:])

    def test_atomic_teacher_resume_conflict_and_corruption(self):
        row={'sid':'sample','fingerprint':'inputsha'};db=connection(self.cfg,'metric')
        a=np.stack((np.full(SIZE,10,np.float32),np.ones(SIZE,np.float32)))
        self.assertFalse(already_done(self.cfg,'metric',row,'recipe',db))
        commit_record(self.cfg,'metric',row,'recipe',a,db)
        self.assertTrue(already_done(self.cfg,'metric',row,'recipe',db))
        with self.assertRaises(RuntimeError):already_done(self.cfg,'metric',row,'different',db)
        with cache_path(self.cfg,'metric','sample').open('ab') as f:f.write(b'corruption')
        with self.assertRaises(RuntimeError):already_done(self.cfg,'metric',row,'recipe',db)
        db.close()

    def test_checkpoint_rng_optimizer_and_alias(self):
        layer=torch.nn.Linear(2,1);opt=torch.optim.AdamW(layer.parameters())
        loss=layer(torch.ones(1,2)).square().sum();loss.backward();opt.step()
        rng=capture_rng();expected=torch.rand(3);restore_rng(rng)
        payload={'epoch':2,'progress':{'global_step':3},'rng':rng,'model':layer.state_dict(),'optimizer':opt.state_dict(),
                 'state':{'epoch':2,'cursor':7,'totals':{'total':5.},'seen':28}}
        path=self.root/'checkpoints'/'last.pth';commit_checkpoint(path,payload)
        restored=load_checkpoint(path);restore_rng(restored['rng'])
        torch.testing.assert_close(torch.rand(3),expected)
        self.assertEqual(restored['state']['cursor'],7)
        alias_checkpoint(path,path.with_name('best_policy.pth'))
        self.assertEqual(load_checkpoint(path.with_name('best_policy.pth'))['epoch'],2)

    def test_untrusted_zip_path_rejected(self):
        from acquire import extract_selected
        zpath=self.root/'bad.zip'
        with zipfile.ZipFile(zpath,'w') as z:z.writestr('../escape.txt','bad')
        with self.assertRaises(ValueError):extract_selected(zpath,self.root/'extract')
        self.assertFalse((self.root/'escape.txt').exists())

    def test_relative_empty_confidence_rejected(self):
        arrays=np.stack((np.linspace(-1,1,SIZE[0]*SIZE[1],dtype=np.float32).reshape(SIZE),
                         np.zeros(SIZE,np.float32)))
        with self.assertRaises(RuntimeError):validate_arrays(arrays,'relative')
        arrays[1].fill(1)
        validate_arrays(arrays,'relative')

    def test_relative_prediction_never_reads_gt_or_sparse(self):
        self.fixtures();row=FullKITTI(self.cfg,'train').rows[0]
        row['gt']['path']='not-a-real-gt.png';row['sparse']['path']='not-a-real-sparse.png'
        class RGBOnly:
            def predict(inner,rgb):
                self.assertEqual(rgb.shape,(*SIZE,3))
                self.assertEqual(rgb.dtype,np.uint8)
                return 'rgb-only'
        self.assertEqual(predict_row(self.cfg,'relative',row,RGBOnly()),'rgb-only')

    def test_full_teacher_audit_exact_train_ids_only(self):
        from teachers import teacher_recipe
        from utils import freeze
        self.fixtures();self.cfg['relative_long_side']=1232
        train=FullKITTI(self.cfg,'train')
        metric=np.stack((np.full(SIZE,10,np.float32),np.ones(SIZE,np.float32)))
        relative=np.stack((np.linspace(-1,1,SIZE[0]*SIZE[1],dtype=np.float32).reshape(SIZE),
                           np.ones(SIZE,np.float32)))
        for role,arrays in (('metric',metric),('relative',relative)):
            recipe=teacher_recipe(self.cfg,role)
            freeze(Path(self.cfg['teacher_root'])/role/'recipe.json',recipe)
            db=connection(self.cfg,role)
            for row in train.rows:commit_record(self.cfg,role,row,identity(recipe),arrays,db)
            db.close()
        report=audit(self.cfg)
        self.assertTrue(report['passed'])
        self.assertEqual(report['teachers']['relative']['cached_val'],0)
        self.assertEqual(tuple(FullKITTI(self.cfg,'train',True)[0]['relative'].shape),(1,*SIZE))
        # A committed validation record is forbidden, even if its tensor looks valid.
        db=connection(self.cfg,'metric');val=FullKITTI(self.cfg,'val').rows[0]
        commit_record(self.cfg,'metric',val,identity(teacher_recipe(self.cfg,'metric')),metric,db)
        db.close()
        with self.assertRaises(RuntimeError):audit(self.cfg)
        self.assertFalse(read_json(Path(self.cfg['work'])/'data_gate.json')['passed'])

    def test_midbatch_resume_matches_uninterrupted_toy_updates(self):
        """Bitwise CPU helper contract, NOT a claim of bitwise CUDA kernels."""
        random.seed(42);np.random.seed(42);torch.manual_seed(42)
        model=torch.nn.Sequential(torch.nn.Linear(2,4),torch.nn.Dropout(.1),torch.nn.Linear(4,1))
        optimizer=torch.optim.AdamW(model.parameters(),lr=.002)
        initial_model=copy.deepcopy(model.state_dict());initial_rng=capture_rng()
        def updates(net,opt,start=0):
            for indices in ResumableBatches(23,4,42,epoch=3,start=start):
                x=torch.tensor([[i/23,(i%3)/3] for i in indices])
                x=x+torch.rand_like(x)*.05+random.random()*.01+np.random.random()*.01
                loss=(net(x)-x.sum(1,keepdim=True)).square().mean()
                loss.backward();opt.step();opt.zero_grad(set_to_none=True)
                yield len(indices)
        uninterrupted=list(updates(model,optimizer))
        expected=copy.deepcopy(model.state_dict());expected_rng=capture_rng()
        model.load_state_dict(initial_model);optimizer=torch.optim.AdamW(model.parameters(),lr=.002)
        restore_rng(initial_rng);iterator=updates(model,optimizer)
        next(iterator);next(iterator)
        path=self.root/'resume'/'last.pth'
        commit_checkpoint(path,{'epoch':3,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                                'rng':capture_rng(),'state':{'cursor':2}})
        restored=load_checkpoint(path)
        resumed=torch.nn.Sequential(torch.nn.Linear(2,4),torch.nn.Dropout(.1),torch.nn.Linear(4,1))
        resumed_opt=torch.optim.AdamW(resumed.parameters(),lr=.002)
        resumed.load_state_dict(restored['model']);resumed_opt.load_state_dict(restored['optimizer'])
        restore_rng(restored['rng']);remaining=list(updates(resumed,resumed_opt,restored['state']['cursor']))
        self.assertEqual(remaining,uninterrupted[2:])
        for key,value in resumed.state_dict().items():
            torch.testing.assert_close(value,expected[key],rtol=0,atol=0)
        self.assertTrue(torch.equal(capture_rng()['torch'],expected_rng['torch']))

    def test_cuda_compatibility_patch_rejects_unknown_source(self):
        from extension_compat import modernize_text
        with self.assertRaises(RuntimeError):modernize_text('unknown Tensor.type() source')

    def test_launcher_plan_is_detached_from_tracked_config(self):
        from types import SimpleNamespace
        from launch import launch_plan,cuda_arch,PACKAGE
        original=sha256(PACKAGE/'config.json')
        args=SimpleNamespace(storage_root=str(self.root/'host-ssd'),accept_kitti_license=True,cuda_arch='8.0')
        cfg,env,path=launch_plan(args)
        self.assertTrue(cfg['accept_kitti_license'])
        self.assertEqual(cfg['initialization'],'fresh_imagenet_encoder')
        self.assertEqual(cfg['training']['epochs'],40)
        self.assertEqual(env['DATA_DIR'],str((self.root/'host-ssd'/'data').resolve()))
        self.assertEqual(env['CONFIG_FILE'],str(path))
        self.assertFalse((self.root/'host-ssd').exists())
        self.assertEqual(sha256(PACKAGE/'config.json'),original)
        self.assertEqual(cuda_arch('8.0;9.0+PTX'),'8.0;9.0+PTX')
        with self.assertRaises(ValueError):cuda_arch('8.0; rm -rf')

    def test_launcher_requires_explicit_license_acknowledgement(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from launch import launch_plan
        with patch('launch.read_json',return_value={'accept_kitti_license':False}):
            with self.assertRaises(ValueError):
                launch_plan(SimpleNamespace(accept_kitti_license=False,storage_root=None,cuda_arch='8.0'))

    def test_model_adapter_preserves_v11_3_math_and_gradient(self):
        from adapter import ModelAdapter,model_config
        cfg=read_json(Path(__file__).with_name('config.json'))
        cfg['training'].update(amp='fp32',channels_last=False)
        model=ModelAdapter(cfg,torch.device('cpu'));model.network.eval()
        torch.manual_seed(7)
        rgb=torch.rand(1,3,32,64);mask=(torch.rand(1,1,32,64)<.05).float();sparse=10*mask
        k=torch.tensor([[[40.,0.,32.],[0.,40.,16.],[0.,0.,1.]]])
        gt=torch.full_like(sparse,12.);valid=torch.zeros_like(mask);valid[:,:,10:20,20:40]=1
        batch={'rgb':rgb,'sparse':sparse,'mask':mask,'K':k,'gt':gt*valid,'gt_mask':valid,
               'teacher':gt.clone(),'confidence':torch.ones_like(gt),
               'relative':torch.linspace(-1,1,64)[None,None,None].expand(1,1,32,64).clone(),
               'relative_confidence':torch.ones_like(gt)}
        output=model.forward(batch)
        self.assertEqual(sum(p.numel() for p in model.network.parameters()),584845)
        self.assertEqual(tuple(output['D_full'].shape),(1,1,32,64))
        self.assertEqual(model.network.fine_node.last_solver_report['nfe'],2)
        self.assertEqual(model.network.dynamics.last_solver_report['terminal_time'],1.)
        loss,stats=model.loss(output,batch,mask*0,3.)
        self.assertTrue(torch.isfinite(loss));loss.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.network.parameters()))
        self.assertGreater(float(stats['kd_coverage']),0)
        self.assertGreater(float(stats['relative_pair_coverage']),0)


if __name__=='__main__':unittest.main(verbosity=2)
