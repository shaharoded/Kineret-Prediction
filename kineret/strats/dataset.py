import pickle
import numpy as np
from tqdm import tqdm
import torch
from kineret.strats.model_utils import CycleIndex
import os


class Dataset:
    def __init__(self, args) -> None:
        # read data
        filepath = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed', args.dataset+'.pkl')
        )
        loaded = pickle.load(open(filepath,'rb'))
        if len(loaded)==6:
            data, oc, train_ids, val_ids, test_ids, metadata = loaded
        else:
            data, oc, train_ids, val_ids, test_ids = loaded
            metadata = {}
        args.outcome_names = metadata.get('outcome_names', ['in_hospital_mortality'])
        args.num_labels = len(args.outcome_names)
        args.static_varis = metadata.get('static_varis', None)
        args.input_hours = metadata.get('input_hours', 48.0)
        args.horizon_hours = metadata.get('horizon_hours', 288.0)
        # Length-of-stay regression target: z-scored hours from admission to RELEASE.
        # Patients who died or have no terminal event are masked out of the LoS loss.
        args.los_mean = float(metadata.get('los_mean', 0.0))
        args.los_std  = float(metadata.get('los_std',  1.0)) or 1.0
        # Per-outcome time-to-event normalisation, fitted on train positives by
        # the preprocessor. Feeds the time head so every model in the benchmark
        # answers the same three questions: will it happen, when, and how long
        # is the stay.
        args.time_mean = metadata.get('time_mean', {})
        args.time_std  = metadata.get('time_std', {})
        args.sample_patient = metadata.get('sample_patient', {})
        self.args = args
        self.first_hours = None
        run, totalruns = list(map(int, args.run.split('o')))
        num_train = int(np.ceil(args.train_frac*len(train_ids)))
        start = int(np.linspace(0,len(train_ids)-num_train,totalruns)[run-1])
        train_ids = train_ids[start:start+num_train]
        num_val = int(np.ceil(args.train_frac*len(val_ids)))
        start = int(np.linspace(0,len(val_ids)-num_val,totalruns)[run-1])
        val_ids = val_ids[start:start+num_val]
        args.logger.write('\nPreparing dataset '+args.dataset)
        static_varis = self.get_static_varis(args.dataset)
        # keep variables seen in training set only
        train_variables = data.loc[data.ts_id.isin(train_ids)].variable.unique()
        all_variables = data.variable.unique()
        delete_variables = np.setdiff1d(all_variables, train_variables)
        args.logger.write('Removing variables not in training set: '+str(delete_variables))
        data = data.loc[data.variable.isin(train_variables)]
        curr_ids = data.ts_id.unique()
        train_ids = np.intersect1d(train_ids, curr_ids)
        val_ids = np.intersect1d(val_ids, curr_ids)
        test_ids = np.intersect1d(test_ids, curr_ids)
        args.logger.write('# train, val, test TS: '+str([len(train_ids), len(val_ids), len(test_ids)]))
        sup_ts_ids = np.concatenate((train_ids, val_ids, test_ids))
        ts_id_to_ind = {ts_id:i for i,ts_id in enumerate(sup_ts_ids)}
        data = data.loc[data.ts_id.isin(sup_ts_ids)]
        data['ts_ind'] = data['ts_id'].map(ts_id_to_ind)

        # Get y and N
        oc = oc.loc[oc.ts_id.isin(sup_ts_ids)].copy()
        oc['ts_ind'] = oc['ts_id'].map(ts_id_to_ind)
        oc = oc.sort_values(by='ts_ind')
        y = np.array(oc[args.outcome_names]).astype(float)
        if y.ndim==1:
            y = y.reshape(-1, 1)
        first_hour_cols = [o+'__first_hour' for o in args.outcome_names]
        if all(c in oc.columns for c in first_hour_cols):
            self.first_hours = np.array(oc[first_hour_cols]).astype(float)
        # Time-to-event targets: z-scored first-occurrence hours per outcome.
        # `time_mask` is 1 only where the outcome actually occurs -- a negative
        # has no "when", so it contributes nothing to the regression.
        if self.first_hours is not None:
            means = np.array([args.time_mean.get(o, 0.0) for o in args.outcome_names])
            stds  = np.array([args.time_std.get(o, 1.0) or 1.0 for o in args.outcome_names])
            valid = np.isfinite(self.first_hours)
            filled = np.where(valid, self.first_hours, means[None, :])
            self.time_target_norm = ((filled - means[None, :]) / stds[None, :]).astype(np.float32)
            self.time_mask = valid.astype(np.float32)
        else:
            self.time_target_norm = np.zeros_like(y, dtype=np.float32)
            self.time_mask = np.zeros_like(y, dtype=np.float32)
        # Length-of-stay target — raw hours (NaN where missing), plus a mask
        # and a z-scored copy aligned to the same patient ordering as `y`.
        if 'length_of_stay_hours' in oc.columns:
            los_raw = oc['length_of_stay_hours'].to_numpy(dtype=float)
            los_valid = np.isfinite(los_raw)
            self.los_hours_raw = los_raw
            self.los_mask = los_valid.astype(np.float32)
            los_filled = np.where(los_valid, los_raw, args.los_mean)
            self.los_target_norm = ((los_filled - args.los_mean) / args.los_std).astype(np.float32)
        else:
            self.los_hours_raw = np.full(len(oc), np.nan, dtype=float)
            self.los_mask = np.zeros(len(oc), dtype=np.float32)
            self.los_target_norm = np.zeros(len(oc), dtype=np.float32)
        N = len(sup_ts_ids)

        # To save
        self.N = N
        self.y = y
        self.static_varis = static_varis
        self.ind_to_ts_id = {i:ts_id for ts_id,i in ts_id_to_ind.items()}
        self.splits = {'train':[ts_id_to_ind[i] for i in train_ids],
                       'val':[ts_id_to_ind[i] for i in val_ids],
                       'test':[ts_id_to_ind[i] for i in test_ids]}
        self.splits['eval_train'] = self.splits['train'][:2000]
        self.train_cycler = CycleIndex(self.splits['train'], args.train_batch_size)
        num_train = len(train_ids)
        num_train_pos = y[self.splits['train']].sum(axis=0)
        args.pos_class_weight = (num_train-num_train_pos)/np.maximum(num_train_pos, 1)
        args.logger.write('pos class weight: '+str(args.pos_class_weight))
        args.logger.write('% pos class in train, val, test splits: '
                          +str([num_train_pos/num_train,
                                y[self.splits['val']].sum(axis=0)/len(val_ids),
                                y[self.splits['test']].sum(axis=0)/len(test_ids)]))

        # Get static data with missingness indicator.
        data = self.get_static_data(data)

        # Trim to the per-sample observation cap.
        data = data.sample(frac=1)
        data = data.groupby('ts_id').head(args.max_obs)

        # Per-variable z-scoring, fitted on TRAIN samples only.
        means_stds = data.loc[data.ts_id.isin(train_ids)].groupby(
                            'variable').agg({'value':['mean', 'std']})
        means_stds.columns = [col[1] for col in means_stds.columns]
        means_stds.loc[means_stds['std']==0, 'std'] = 1
        # `max_minute` scales time into [-1, 1]. It spans the longest training
        # window, so a short-window sample's timestamps sit on the same scale as
        # a long one's and the model can tell how much history it was given.
        max_minute = data['minute'].max()
        data = data.merge(means_stds.reset_index(), on='variable', how='left')
        data['value'] = (data['value']-data['mean'])/data['std']

        # prepare time series inputs
        variables = data.variable.unique()
        var_to_ind = {v:i for i,v in enumerate(variables)}
        V = len(variables)
        args.V = V
        args.logger.write('# TS variables: '+str(V))
        values = [[] for i in range(N)]
        times = [[] for i in range(N)]
        varis = [[] for i in range(N)]
        data['minute'] = data['minute']/max_minute*2-1
        for row in data.itertuples():
            values[row.ts_ind].append(row.value)
            times[row.ts_ind].append(row.minute)
            varis[row.ts_ind].append(var_to_ind[row.variable])
        self.values, self.times, self.varis = values, times, varis

    def get_static_varis(self, dataset):
        """Purpose: Static feature names, always supplied by the pickle metadata."""
        static_varis = getattr(self.args, 'static_varis', None)
        if static_varis is None:
            raise ValueError(
                "[Dataset] The pickle carries no `static_varis` metadata. "
                "Rebuild it with kineret.strats.preprocess.build_pickle.")
        return list(static_varis)

    def get_static_data(self, data):
                # Get static data with missingness indicator.
        static_ii = data.variable.isin(self.static_varis)
        static_data = data.loc[static_ii]
        data = data.loc[~static_ii] # remove static vars from data
        static_var_to_ind = {v:i for i,v in enumerate(self.static_varis)}
        D = len(static_var_to_ind)
        demo = np.zeros((self.N, D))
        for row in tqdm(static_data.itertuples(), desc='static features', leave=False):
            demo[row.ts_ind, static_var_to_ind[row.variable]] = row.value
        # Normalize static data.
        train_ind = self.splits['train']
        means = demo[train_ind].mean(axis=0, keepdims=True)
        stds = demo[train_ind].std(axis=0, keepdims=True)
        stds = (stds==0) + (stds>0)*stds
        demo = (demo-means)/stds
        self.args.logger.write('# static features: '+str(D))
        # to save
        self.demo = demo
        self.args.D = D
        return data


    def get_batch(self, ind=None):
        """
        Purpose: One training or evaluation batch of sparse observation triplets.

        Args:
            ind (array-like|None): Sample indices; None draws the next training batch.

        Returns:
            dict: values / times / varis / obs_mask / demo / labels + the onset
                  and length-of-stay supervision tensors.
        """
        if ind is None:
            ind = self.train_cycler.get_batch_ind()
        return self.get_batch_strats(ind)

    def get_batch_strats(self, ind):
        demo = torch.FloatTensor(self.demo[ind]) # N,D
        num_obs = [len(self.values[i]) for i in ind]
        max_obs = max(num_obs)
        pad_lens = max_obs-np.array(num_obs)
        values = [self.values[i]+[0]*(l) for i,l in zip(ind,pad_lens)]
        times = [self.times[i]+[0]*(l) for i,l in zip(ind,pad_lens)]
        varis = [self.varis[i]+[0]*(l) for i,l in zip(ind,pad_lens)]
        values, times = torch.FloatTensor(values), torch.FloatTensor(times)
        varis = torch.IntTensor(varis)
        obs_mask = [[1]*l1+[0]*l2 for l1,l2 in zip(num_obs,pad_lens)]
        obs_mask = torch.IntTensor(obs_mask)
        return {'values':values, 'times':times, 'varis':varis,
                'obs_mask':obs_mask, 'demo':demo,
                'labels':torch.FloatTensor(self.y[ind]),
                'los_target_norm':torch.FloatTensor(self.los_target_norm[ind]),
                'los_mask':torch.FloatTensor(self.los_mask[ind]),
                'time_target_norm':torch.FloatTensor(self.time_target_norm[ind]),
                'time_mask':torch.FloatTensor(self.time_mask[ind])}
