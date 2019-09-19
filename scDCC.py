import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from layers import ZINBLoss, MeanAct, DispAct

def buildNetwork(layers, type, activation="relu"):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i-1], layers[i]))
        if activation=="relu":
            net.append(nn.ReLU())
        elif activation=="sigmoid":
            net.append(nn.Sigmoid())
    return nn.Sequential(*net)


class scDCC(nn.Module):
    def __init__(self, input_dim, z_dim, n_clusters, encodeLayer=[], decodeLayer=[], 
            activation="relu", sigma=1., alpha=1., gamma=1.):
        super(scDCC, self).__init__()
        self.z_dim = z_dim
        self.n_clusters = n_clusters
        self.activation = activation
        self.sigma = sigma
        self.alpha = alpha
        self.gamma = gamma
        self.encoder = buildNetwork([input_dim]+encodeLayer, type="encode", activation=activation)
        self.decoder = buildNetwork([z_dim]+decodeLayer, type="decode", activation=activation)
        self._enc_mu = nn.Linear(encodeLayer[-1], z_dim)
        self._dec_mean = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), MeanAct())
        self._dec_disp = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), DispAct())
        self._dec_pi = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), nn.Sigmoid())

        self.mu = Parameter(torch.Tensor(n_clusters, z_dim))
        self.zinb_loss = ZINBLoss().cuda()
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)
    
    def soft_assign(self, z):
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.mu)**2, dim=2) / self.alpha)
        q = q**((self.alpha+1.0)/2.0)
        q = (q.t() / torch.sum(q, dim=1)).t()
        return q
    
    def target_distribution(self, q):
        p = q**2 / q.sum(0)
        return (p.t() / p.sum(1)).t()
    
    def forward(self, x):
        h = self.encoder(x+torch.randn_like(x) * self.sigma)
        z = self._enc_mu(h)
        h = self.decoder(z)
        _mean = self._dec_mean(h)
        _disp = self._dec_disp(h)
        _pi = self._dec_pi(h)

        h0 = self.encoder(x)
        z0 = self._enc_mu(h0)
        q = self.soft_assign(z0)
        return z0, q, _mean, _disp, _pi
    
    def encodeBatch(self, X, batch_size=256):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        
        encoded = []
        self.eval()
        num = X.shape[0]
        num_batch = int(math.ceil(1.0*X.shape[0]/batch_size))
        for batch_idx in range(num_batch):
            xbatch = X[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
            inputs = Variable(xbatch)
            z,_, _, _, _ = self.forward(inputs)
            encoded.append(z.data)

        encoded = torch.cat(encoded, dim=0)
        return encoded
    
    def cluster_loss(self, p, q):
        def kld(target, pred):
            return torch.mean(torch.sum(target*torch.log(target/(pred+1e-6)), dim=-1))
        kldloss = kld(p, q)
        return self.gamma*kldloss
    
    def pairwise_loss(self, p1, p2, cons_type):
        if cons_type == "ML":
            ml_loss = torch.mean(-torch.log(torch.sum(p1 * p2, dim=1)))
            return ml_loss
        else:
            cl_loss = torch.mean(-torch.log(1.0 - torch.sum(p1 * p2, dim=1)))
            return cl_loss
    
    def global_size_loss(self, p, cons_detail):
        m_p = torch.mean(p, dim=0)
        m_p = m_p / torch.sum(m_p)
        return torch.sum((m_p-cons_detail)*(m_p-cons_detail))

    def difficulty_loss(self, q, mask):
        mask = mask.unsqueeze_(-1)
        mask = mask.expand(q.shape[0], q.shape[1])
        mask_q = q * mask
        diff_loss = -torch.norm(mask_q, 2)
        penalty_degree = 0.1
        return penalty_degree * diff_loss
    
    def triplet_loss(self, anchor, positive, negative, margin_constant):
        # loss = max(d(anchor, negative) - d(anchor, positve) + margin, 0), margin > 0
        # d(x, y) = q(x) * q(y)
        negative_dis = torch.sum(anchor * negative, dim=1)
        positive_dis = torch.sum(anchor * positive, dim=1)
        margin = margin_constant * torch.ones(negative_dis.shape).cuda()
        diff_dis = negative_dis - positive_dis
        penalty = diff_dis + margin
        triplet_loss = 1*torch.max(penalty, torch.zeros(negative_dis.shape).cuda())

        return torch.mean(triplet_loss)
    
    def pretrain_autoencoder(self, x, raw_counts, size_factor, batch_size=256, lr=0.001, epochs=400, ae_save=True, ae_weights='AE_weights.pth.tar'):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        dataset = TensorDataset(torch.Tensor(x), torch.Tensor(raw_counts), torch.Tensor(size_factor))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print("Pretraining stage")
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, amsgrad=True)
        for epoch in range(epochs):
            for batch_idx, (x_batch, x_raw_batch, sf_batch) in enumerate(dataloader):
                x_tensor = Variable(x_batch).cuda()
                x_raw_tensor = Variable(x_raw_batch).cuda()
                sf_tensor = Variable(sf_batch).cuda()
                _, _, mean_tensor, disp_tensor, pi_tensor = self.forward(x_tensor)
                loss = self.zinb_loss(x=x_raw_tensor, mean=mean_tensor, disp=disp_tensor, pi=pi_tensor, scale_factor=sf_tensor)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print('Pretrain epoch [{}/{}], ZINB loss:{:.4f}'.format(batch_idx+1, epoch+1, loss.item()))
        
        if ae_save:
            torch.save({'ae_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, ae_weights)

    def save_checkpoint(self, state, index, filename):
        newfilename = os.path.join(filename, 'FTcheckpoint_%d.pth.tar' % index)
        torch.save(state, newfilename)

    def fit(self, X, X_raw, sf, anchor=np.array([]), positive=np.array([]), negative=np.array([]), 
            ml_ind1=np.array([]), ml_ind2=np.array([]), cl_ind1=np.array([]), cl_ind2=np.array([]), 
            mask=np.array([]), use_global=False, ml_p=1., cl_p=1., y=None, lr=1., batch_size=256, 
            num_epochs=10, update_interval=1, tol=1e-3, save_dir=""):
        '''X: tensor data'''
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        print("Clustering stage")
        X = torch.tensor(X).cuda()
        X_raw = torch.tensor(X_raw).cuda()
        sf = torch.tensor(sf).cuda()
        mask = torch.zeros(X.shape[0]).cuda()
        optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, rho=.95)

        print("Initializing cluster centers with kmeans.")
        kmeans = KMeans(self.n_clusters, n_init=20)
        data = self.encodeBatch(X)
        self.y_pred = kmeans.fit_predict(data.data.cpu().numpy())
        self.y_pred_last = self.y_pred
        self.mu.data.copy_(torch.Tensor(kmeans.cluster_centers_))
        if y is not None:
            acc = np.round(cluster_acc(y, self.y_pred), 5)
            nmi = np.round(metrics.normalized_mutual_info_score(y, self.y_pred), 5)
            ari = np.round(metrics.adjusted_rand_score(y, self.y_pred), 5)
            print('Initializing k-means: ACC= %.4f, NMI= %.4f, ARI= %.4f' % (acc, nmi, ari))
        
        self.train()
        num = X.shape[0]
        num_batch = int(math.ceil(1.0*X.shape[0]/batch_size))
        ml_num_batch = int(math.ceil(1.0*ml_ind1.shape[0]/batch_size))
        cl_num_batch = int(math.ceil(1.0*cl_ind1.shape[0]/batch_size))
        tri_num_batch = int(math.ceil(1.0*anchor.shape[0]/batch_size))
        cl_num = cl_ind1.shape[0]
        ml_num = ml_ind1.shape[0]
        tri_num = anchor.shape[0]

        final_acc, final_nmi, final_ari, final_epoch = 0, 0, 0, 0
        update_ml = 1
        update_cl = 1
        update_triplet = 1

        for epoch in range(num_epochs):
            if epoch%update_interval == 0:
                # update the targe distribution p
                latent = self.encodeBatch(X)
                q = self.soft_assign(latent)
                p = self.target_distribution(q).data

                # evalute the clustering performance
                self.y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
                if use_global:
                    y_dict = collections.defaultdict(list)
                    ind1, ind2 = [], []
                    for i in range(self.y_pred.shape[0]):
                        y_dict[self.y_pred[i]].append(i)
                    for key in y_dict.keys():
                        if y is not None:
                            print("predicted class: ", key, " total: ", len(y_dict[key]))
                            #, " mapped index(ground truth): ", np.bincount(y[y_dict[key]]).argmax())

                if y is not None:
                    final_acc = acc = np.round(cluster_acc(y, self.y_pred), 5)
                    final_nmi = nmi = np.round(metrics.normalized_mutual_info_score(y, self.y_pred), 5)
                    final_epoch = ari = np.round(metrics.adjusted_rand_score(y, self.y_pred), 5)
                    print('Clustering   %d: ACC= %.4f, NMI= %.4f, ARI= %.4f' % (epoch+1, acc, nmi, ari))

                # save current model
                if (epoch>0 and delta_label < tol) or epoch%10 == 0:
                    self.save_checkpoint({'epoch': epoch+1,
                            'state_dict': self.state_dict(),
                            'mu': self.mu,
                            'p': p,
                            'q': q,
                            'y_pred': self.y_pred,
                            'y_pred_last': self.y_pred_last,
                            'y': y
                            }, epoch+1, filename=save_dir)

                # check stop criterion
                delta_label = np.sum(self.y_pred != self.y_pred_last).astype(np.float32) / num
                self.y_pred_last = self.y_pred
                if epoch>0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print("Reach tolerance threshold. Stopping training.")
                    break


            # train 1 epoch for clustering loss
            train_loss = 0.0
            recon_loss_val = 0.0
            cluster_loss_val = 0.0
            instance_constraints_loss_val = 0.0
            global_loss_val = 0.0
            for batch_idx in range(num_batch):
                xbatch = X[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                xrawbatch = X_raw[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                sfbatch = sf[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                pbatch = p[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                mask_batch = mask[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                optimizer.zero_grad()
                inputs = Variable(xbatch)
                rawinputs = Variable(xrawbatch)
                sfinputs = Variable(sfbatch)
                target = Variable(pbatch)
                cons_detail = np.repeat(1.0/float(self.n_clusters), self.n_clusters)
                global_cons = torch.from_numpy(cons_detail).float().to("cuda")

                z, qbatch, meanbatch, dispbatch, pibatch = self.forward(inputs)
                if use_global == False:
                    cluster_loss = self.cluster_loss(target, qbatch)
                    recon_loss = self.zinb_loss(rawinputs, meanbatch, dispbatch, pibatch, sfinputs)
                    loss = cluster_loss + recon_loss
                    loss.backward()
                    optimizer.step()
                    cluster_loss_val += cluster_loss.data * len(inputs)
                    recon_loss_val += recon_loss.data * len(inputs)
                    train_loss = cluster_loss_val + recon_loss_val + instance_constraints_loss_val
                else:
                    cluster_loss = self.cluster_loss(target, qbatch)
                    recon_loss = self.zinb_loss(rawinputs, meanbatch, dispbatch, pibatch, sfinputs)
                    global_loss = self.global_size_loss(qbatch, global_cons)
                    loss = cluster_loss + recon_loss + global_loss
                    loss.backward()
                    optimizer.step()
                    cluster_loss_val += cluster_loss.data * len(inputs)
                    recon_loss_val += recon_loss.data * len(inputs)
                    train_loss = cluster_loss_val + recon_loss_val


            if instance_constraints_loss_val != 0.0:
                print("#Epoch %3d: Total: %.4f Clustering Loss: %.4f ZINB Loss: %.4f Instance Difficulty Loss: %.4f"% (
                    epoch + 1, train_loss / num, cluster_loss_val / num, recon_loss_val / num, instance_constraints_loss_val / num))
            elif global_loss_val != 0.0 and use_global:
                print("#Epoch %3d: Total: %.4f Clustering Loss: %.4f ZINB Loss: %.4f Global Loss: %.4f"% (
                    epoch + 1, train_loss / num + global_loss_val/num_batch, cluster_loss_val / num, recon_loss_val / num, global_loss_val / num_batch))
            else:
                print("#Epoch %3d: Total: %.4f Clustering Loss: %.4f ZINB Loss: %.4f" % (
                    epoch + 1, train_loss / num, cluster_loss_val / num, recon_loss_val / num))

            ml_loss = 0.0
            if epoch % update_ml == 0:
                for ml_batch_idx in range(ml_num_batch):
                    px1 = X[ml_ind1[ml_batch_idx*batch_size : min(ml_num, (ml_batch_idx+1)*batch_size)]]
                    pxraw1 = X_raw[ml_ind1[ml_batch_idx*batch_size : min(ml_num, (ml_batch_idx+1)*batch_size)]]
                    sf1 = sf[ml_ind1[ml_batch_idx*batch_size : min(ml_num, (ml_batch_idx+1)*batch_size)]]
                    px2 = X[ml_ind2[ml_batch_idx*batch_size : min(ml_num, (ml_batch_idx+1)*batch_size)]]
                    sf2 = sf[ml_ind2[ml_batch_idx*batch_size : min(ml_num, (ml_batch_idx+1)*batch_size)]]
                    pxraw2 = X_raw[ml_ind2[ml_batch_idx*batch_size : min(ml_num, (ml_batch_idx+1)*batch_size)]]
                    pbatch1 = p[ml_ind1[ml_batch_idx*batch_size : min(ml_num, (ml_batch_idx + 1)*batch_size)]]
                    pbatch2 = p[ml_ind2[ml_batch_idx*batch_size : min(ml_num, (ml_batch_idx+1)*batch_size)]]
                    optimizer.zero_grad()
                    inputs1 = Variable(px1)
                    rawinputs1 = Variable(pxraw1)
                    sfinput1 = Variable(sf1)
                    inputs2 = Variable(px2)
                    rawinputs2 = Variable(pxraw2)
                    sfinput2 = Variable(sf2)
                    target1 = Variable(pbatch1)
                    target2 = Variable(pbatch2)
                    z1, q1, mean1, disp1, pi1 = self.forward(inputs1)
                    z2, q2, mean2, disp2, pi2 = self.forward(inputs2)
                    loss = (ml_p*self.pairwise_loss(q1, q2, "ML")+self.zinb_loss(rawinputs1, mean1, disp1, pi1, sfinput1) + self.zinb_loss(rawinputs2, mean2, disp2, pi2, sfinput2))
                    # 0.1 for mnist/reuters, 1 for fashion, the parameters are tuned via grid search on validation set
                    ml_loss += loss.data
                    loss.backward()
                    optimizer.step()

            cl_loss = 0.0
            if epoch % update_cl == 0:
                for cl_batch_idx in range(cl_num_batch):
                    px1 = X[cl_ind1[cl_batch_idx*batch_size : min(cl_num, (cl_batch_idx+1)*batch_size)]]
                    px2 = X[cl_ind2[cl_batch_idx*batch_size : min(cl_num, (cl_batch_idx+1)*batch_size)]]
                    pbatch1 = p[cl_ind1[cl_batch_idx*batch_size : min(cl_num, (cl_batch_idx + 1)*batch_size)]]
                    pbatch2 = p[cl_ind2[cl_batch_idx*batch_size : min(cl_num, (cl_batch_idx+1)*batch_size)]]
                    optimizer.zero_grad()
                    inputs1 = Variable(px1)
                    inputs2 = Variable(px2)
                    target1 = Variable(pbatch1)
                    target2 = Variable(pbatch2)
                    z1, q1, _, _, _ = self.forward(inputs1)
                    z2, q2, _, _, _ = self.forward(inputs2)
                    loss = cl_p*self.pairwise_loss(q1, q2, "CL")
                    cl_loss += loss.data
                    loss.backward()
                    optimizer.step()

            if ml_num_batch >0 and cl_num_batch > 0:
                print("Pairwise Total:", round(float(ml_loss.cpu()), 2) + float(cl_loss.cpu()), "ML loss", float(ml_loss.cpu()), "CL loss:", float(cl_loss.cpu()))
            triplet_loss = 0.0
            if epoch % update_triplet == 0:
                for tri_batch_idx in range(tri_num_batch):
                    px1 = X[anchor[tri_batch_idx*batch_size : min(tri_num, (tri_batch_idx+1)*batch_size)]]
                    px2 = X[positive[tri_batch_idx*batch_size : min(tri_num, (tri_batch_idx+1)*batch_size)]]
                    px3 = X[negative[tri_batch_idx*batch_size : min(tri_num, (tri_batch_idx+1)*batch_size)]]
                    pbatch1 = p[anchor[tri_batch_idx*batch_size : min(tri_num, (tri_batch_idx + 1)*batch_size)]]
                    pbatch2 = p[positive[tri_batch_idx*batch_size : min(tri_num, (tri_batch_idx+1)*batch_size)]]
                    pbatch3 = p[negative[tri_batch_idx*batch_size : min(tri_num, (tri_batch_idx+1)*batch_size)]]
                    optimizer.zero_grad()
                    inputs1 = Variable(px1)
                    inputs2 = Variable(px2)
                    inputs3 = Variable(px3)
                    target1 = Variable(pbatch1)
                    target2 = Variable(pbatch2)
                    target3 = Variable(pbatch3)
                    z1, q1, _, _, _ = self.forward(inputs1)
                    z2, q2, _, _, _ = self.forward(inputs2)
                    z3, q3, _, _, _ = self.forward(inputs3)
                    loss = self.triplet_loss(q1, q2, q3, 0.1)
                    triplet_loss += loss.data
                    loss.backward()
                    optimizer.step()
            if tri_num_batch > 0:
                print("Triplet Loss:", triplet_loss)

        return final_acc, final_nmi, final_ari, final_epoch