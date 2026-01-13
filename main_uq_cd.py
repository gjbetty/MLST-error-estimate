
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.autograd as autograd

import numpy as np
import scipy.io
from tqdm import trange

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# In[ ]:


np.random.seed(123)
torch.manual_seed(123)
torch.set_default_dtype(torch.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print(torch.cuda.get_device_name())

GPU_ENABLED = True
if torch.cuda.is_available():
    try:
        _ = torch.Tensor([0., 0.]).cuda()
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        print('gpu available')
        GPU_ENABLED = True
    except:
        print('gpu not available')
        GPU_ENABLED = False
else:
    print('gpu not available')
    GPU_ENABLED = False

# In[ ]:
class MLP1D(nn.Module):
    def __init__(self, layers, activation=torch.tanh):
        super().__init__()
        self.depth = len(layers) - 1

        self.activation = activation
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(self.depth)])

        for linear in self.linears:
            nn.init.xavier_normal_(linear.weight.data)
            nn.init.zeros_(linear.bias.data)

    def input_encoding(self, x, t,a,b,c,d):
        out = torch.concat([x, t,a,b,c,d], dim=1)
        return out

    def forward(self, x, t,a,b,c,d):
        X = self.input_encoding(x, t,a,b,c,d)

        for i in range(self.depth - 1):
            Y = self.linears[i](X)
            X = self.activation(Y)

        return self.linears[-1](X)



# Define MLP with exact periodicity
class MLP(nn.Module):
    def __init__(self,layers,L=1.0,M = 1,activation = torch.tanh):
        super().__init__()
        self.depth = len(layers) -1

        self.omega = nn.Parameter(torch.tensor(2*torch.pi/L),requires_grad = False)
        self.k = nn.Parameter(torch.arange(M + 1),requires_grad = False)

        self.activation = activation
        self.linears = nn.ModuleList([nn.Linear(layers[i],layers[i+1]) for i in range(self.depth)])

        for linear in self.linears:
            nn.init.xavier_normal_(linear.weight.data)
            nn.init.zeros_(linear.bias.data)

    def input_encoding(self,x,t):
        out = torch.concat([t,torch.cos(self.k*self.omega*x),torch.sin(self.k[1:]*self.omega*x)],dim = 1)
        return out

    def forward(self,x,t):
        X = self.input_encoding(x,t)

        for i in range(self.depth-1):
            Y = self.linears[i](X)
            X = self.activation(Y)

        return self.linears[-1](X)


# In[ ]:


class ModifiedMLP(nn.Module):
    def __init__(self,layers,L=1.0,M=1,activation = torch.tanh):
        super().__init__()
        self.depth = len(layers)-1

        self.omega = nn.Parameter(torch.tensor(2*torch.pi/L),requires_grad = False)
        self.k = nn.Parameter(torch.arange(M + 1),requires_grad = False)

        self.activate = activation

        self.linears = nn.ModuleList([nn.Linear(layers[i],layers[i+1]) for i in range(self.depth)])
        for linear in self.linears:
            nn.init.xavier_normal_(linear.weight.data)
            nn.init.zeros_(linear.bias.data)

        self.U = nn.Linear(layers[0],layers[1])
        nn.init.xavier_normal_(self.U.weight.data)
        self.V = nn.Linear(layers[0],layers[1])
        nn.init.xavier_normal_(self.V.weight.data)

    def input_encodeing(self,x,t):
        out = torch.concat([t,torch.cos(self.k*self.omega*x),torch.sin(self.k[1:]*self.omega*x)],dim = 1)
        return out

    def forward(self,x,t):
        X = self.input_encodeing(x,t)
        U = self.U(X)
        V = self.V(X)

        H = self.linears[0](X)
        for i in range(1,self.depth-1):
            Z = self.linears[i](H)
            H = (1-Z)*U + Z*V

        return self.linears[-1](H)


# In[ ]:


class CasualPINN():
    def __init__(self,layers,M,x_star,y_star,usol,alpha,beta,gamma,kappa,nt,nx,tol):

        self.nt = nt
        self.nx = nx

        # self.u_ref = usol
        # self.u_x_ref = usol_ux
        # self.u_t_ref = usol_ut
        self.u_ref = torch.from_numpy(usol).cuda().reshape(-1,1)
        # self.u_x_ref = torch.from_numpy(usol_ux).cuda().reshape(-1,1)
        # self.u_t_ref = torch.from_numpy(usol_ut).cuda().reshape(-1,1)
         # self.T = t1

        # collection points 100x512
        self.t_r = np.linspace(-1,1,nt)
        self.x_r = np.linspace(-10.0,10.0,nx)
        self.t_0 = -1*np.ones(1)
        self.alpha = 9 #np.linspace(8.7, 9.3, 3)
        self.beta = 0 #np.linspace(-0.4, 0.4, 3)
        self.gamma = np.linspace(0.9, 1.1, 3)
        self.kappa = np.linspace(0.9, 1.1, 3)

        X,T,A,B,C,D = np.meshgrid(self.x_r,self.t_r,self.alpha,self.beta,self.gamma,self.kappa)

        X_ic,T_ic,A_ic,B_ic,C_ic,D_ic = np.meshgrid(x_star,self.t_0,self.alpha,self.beta,self.gamma,self.kappa)

        # self.X_r = torch.tensor(np.concatenate([X.flatten()[:,None],T.flatten()[:,None],A.flatten()[:,None],B.flatten()[:,None],C.flatten()[:,None],D.flatten()[:,None]],axis = 1),requires_grad=True).float().to(device)
        self.X = torch.tensor(X.flatten()[:,None],requires_grad=True)
        self.T = torch.tensor(T.flatten()[:, None], requires_grad=True)
        self.A = torch.tensor(A.flatten()[:, None], requires_grad=False)
        self.B = torch.tensor(B.flatten()[:, None], requires_grad=False)
        self.C = torch.tensor(C.flatten()[:, None], requires_grad=False)
        self.D = torch.tensor(D.flatten()[:, None], requires_grad=False)
        self.X_r = torch.concat([self.X,self.T,self.A,self.B,self.C,self.D],dim = 1).float().to(device)

        # For computing the temporal weight
        self.M = torch.tril(torch.ones(nt,nt),-1).float().to(device)
        self.tol = tol

        # IC 2048,1,3,3,3,3
        # self.x_ic = torch.tensor(X).float().reshape(-1,1)
        # self.t_ic = -1*torch.ones_like(self.x_ic)

        self.X_ic_tensor = torch.tensor(X_ic.flatten()[:,None],requires_grad=True)
        self.T_ic_tensor = torch.tensor(T_ic.flatten()[:, None], requires_grad=True)
        self.A_ic_tensor = torch.tensor(A_ic.flatten()[:, None], requires_grad=False)
        self.B_ic_tensor = torch.tensor(B_ic.flatten()[:, None], requires_grad=False)
        self.C_ic_tensor = torch.tensor(C_ic.flatten()[:, None], requires_grad=False)
        self.D_ic_tensor = torch.tensor(D_ic.flatten()[:, None], requires_grad=False)

        self.X_ic = torch.concat([self.X_ic_tensor,self.T_ic_tensor,self.A_ic_tensor,self.B_ic_tensor,self.C_ic_tensor,self.D_ic_tensor],dim = 1).float().to(device)
        self.Y_ic = torch.tensor(y_star).float().reshape(-1,1).to(device)
        # self.X_ic.requires_grad = True


        Nbc = 256

        t_bc = 2 * np.random.rand(Nbc, 1) - 1
        x_bc_l = -10*np.full([1, 1], 1).astype(np.float64)
        x_bc_r = 10*np.ones([1, 1]).astype(np.float64)
        X_bc_l,T_bc_l,A_bc_l,B_bc_l,C_bc_l,D_bc_l = np.meshgrid(x_bc_l,t_bc,self.alpha,self.beta,self.gamma,self.kappa)
        X_bc_r, T_bc_r, A_bc_r, B_bc_r, C_bc_r, D_bc_r = np.meshgrid(x_bc_r, t_bc, self.alpha, self.beta, self.gamma, self.kappa)


        self.X_bcl_tensor = torch.tensor(X_bc_l.flatten()[:,None],requires_grad=True)
        self.T_bcl_tensor = torch.tensor(T_bc_l.flatten()[:,None],requires_grad=True)
        self.A_bcl_tensor = torch.tensor(A_bc_l.flatten()[:, None], requires_grad=False)
        self.B_bcl_tensor = torch.tensor(B_bc_l.flatten()[:, None], requires_grad=False)
        self.C_bcl_tensor = torch.tensor(C_bc_l.flatten()[:, None], requires_grad=False)
        self.D_bcl_tensor = torch.tensor(D_bc_l.flatten()[:, None], requires_grad=False)


        self.X_bcr_tensor = torch.tensor(X_bc_r.flatten()[:,None],requires_grad=True)
        self.T_bcr_tensor = torch.tensor(T_bc_r.flatten()[:,None],requires_grad=True)
        self.A_bcr_tensor = torch.tensor(A_bc_r.flatten()[:, None], requires_grad=False)
        self.B_bcr_tensor = torch.tensor(B_bc_r.flatten()[:, None], requires_grad=False)
        self.C_bcr_tensor = torch.tensor(C_bc_r.flatten()[:, None], requires_grad=False)
        self.D_bcr_tensor = torch.tensor(D_bc_r.flatten()[:, None], requires_grad=False)

        x_boundary_left = torch.concat([self.X_bcl_tensor,self.T_bcl_tensor,self.A_bcl_tensor,self.B_bcl_tensor,self.C_bcl_tensor,self.D_bcl_tensor], dim=1)
        x_boundary_right = torch.cat([self.X_bcr_tensor,self.T_bcr_tensor,self.A_bcr_tensor,self.B_bcr_tensor,self.C_bcr_tensor,self.D_bcr_tensor], dim=1)
        # x_boundary_left_label = torch.zeros_like(x_boundary_left)
        # x_boundary_right_label = torch.zeros_like(x_boundary_right)
        self.x_bc = torch.cat((x_boundary_left, x_boundary_right), dim=0).float().to(device)
        # self.x_bc.requires_grad = True
        self.u_bc = torch.zeros(Nbc*3*3*2, 1).float().to(device)
        self.u_0 = torch.zeros(Nbc*3*3*2, 1).float().to(device)

        # network
        # self.NN = MLP(layers,L=2,M=M,activation=torch.tanh).to(device)
        # self.NN = MLP(layers,L=2,M=M).to(device)
        self.NN = MLP1D(layers).to(device)

        self.optimizer = optim.Adam(self.NN.parameters())
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer,0.9)

        # Logger
        self.loss_log = []
        self.loss_ics_log = []
        self.loss_bcs_log = []
        self.loss_bcs_ux_log = []
        self.loss_bcs_uxx_log = []
        self.loss_ut_log = []
        self.loss_res_log = []
        self.W_log = []
        self.L_t_log = []
        self.loss_s_u_log = []
        self.loss_s_ux_log = []
        self.loss_s_ut_log = []
        self.loss_s_u_l2_log = []
        self.loss_s_ux_l2_log = []
        self.loss_s_ut_l2_log = []

        # stopping flag
        self.S = False


    def net_u(self,x,t,a,b,c,d):
        u = self.NN(x,t,a,b,c,d)
        # u_m = self.NN(x,t)[:,1]
        # u_m = u_m.unsqueeze(1)
        return u

    def net_ut(self, x, t,a,b,c,d):
        u = self.net_u(x, t,a,b,c,d)
        du = autograd.grad(u, [x, t,a,b,c,d],
                           grad_outputs=torch.ones_like(u),
                           retain_graph=True,
                           create_graph=True)
        u_t = du[1]
        return u_t

    def net_ux(self, x, t,a,b,c,d):
        u = self.net_u(x, t,a,b,c,d)
        du = autograd.grad(u, [x, t,a,b,c,d],
                           grad_outputs=torch.ones_like(u),
                           retain_graph=True,
                           create_graph=True)
        u_x = du[0]

        return u_x

    def net_uxx(self, x, t,a,b,c,d):
        u = self.net_u(x, t,a,b,c,d)
        du = autograd.grad(u, [x, t,a,b,c,d],
                           grad_outputs=torch.ones_like(u),
                           retain_graph=True,
                           create_graph=True)
        u_x = du[0]

        u_xx = autograd.grad(u_x, x,
                             grad_outputs=torch.ones_like(u_x),
                             retain_graph=True,
                             create_graph=True)[0]
        return u_xx

    def net_f(self,x,t,a,b,c,d):
        u = self.net_u(x,t,a,b,c,d)
        du = autograd.grad(u,[x,t,a,b,c,d],
                           grad_outputs=torch.ones_like(u),
                           retain_graph=True,
                           create_graph=True)
        u_x = du[0]
        u_t = du[1]
        # u_tt = autograd.grad(u_t, t,
        #                      grad_outputs=torch.ones_like(u_x),
        #                      retain_graph=True,
        #                      create_graph=True)[0]

        u_xx = autograd.grad(u_x,x,
                             grad_outputs=torch.ones_like(u_x),
                             retain_graph=True,
                             create_graph=True)[0]

        u_xxx = autograd.grad(u_xx, x,
                             grad_outputs=torch.ones_like(u_x),
                             retain_graph=True,
                             create_graph=True)[0]

        # u_xxx = autograd.grad(u_xx,x,
        #                      grad_outputs=torch.ones_like(u_x),
        #                      retain_graph=True,
        #                      create_graph=True)[0]

        res = u_t + c * u_x * u + d * u_xxx

        return res

    # def Mses_u_np(self):
    #     u = self.net_u(self.X_r[:, 0:1], self.X_r[:, 1:2])
    #     a = torch.max(u)
    #     u_np = u.detach().cpu().numpy().reshape(self.u_ref.shape)
    #     mses = np.linalg.norm((u_np - self.u_ref)[:,:]) / np.linalg.norm(self.u_ref)
    #     print('u error is',mses)
    #     return mses

    def Mses_u(self):
        u = self.net_u(self.X_r[:, 0:1], self.X_r[:, 1:2], self.X_r[:, 2:3], self.X_r[:, 3:4], self.X_r[:, 4:5], self.X_r[:, 5:6])
        mses = torch.norm(u - self.u_ref) / torch.norm(self.u_ref)
        # print('u error is', mses)
        return mses

    # def Mses_ux(self):
    #     ux = self.net_ux(self.X_r[:, 0:1], self.X_r[:, 1:2])
    #     mses = torch.norm(ux - self.u_x_ref) / torch.norm(self.u_x_ref)
    #     return mses
    #
    # def Mses_ut(self):
    #     ut = self.net_ut(self.X_r[:, 0:1], self.X_r[:, 1:2])
    #     mses = torch.norm(ut - self.u_t_ref) / torch.norm(self.u_t_ref)
    #     return mses
    #
    # def Mses_u_l2(self):
    #     u = self.net_u(self.X_r[:, 0:1], self.X_r[:, 1:2])
    #     mses = torch.norm(u - self.u_ref)
    #     return mses
    #
    # def Mses_ux_l2(self):
    #     ux = self.net_ux(self.X_r[:, 0:1], self.X_r[:, 1:2])
    #     mses = torch.norm(ux - self.u_x_ref)
    #     return mses
    #
    # def Mses_ut_l2(self):
    #     ut = self.net_ut(self.X_r[:, 0:1], self.X_r[:, 1:2])
    #     mses = torch.norm(ut - self.u_t_ref)
    #     return mses

    def loss_res(self):
        r_pred = self.net_f(self.X_r[:, 0:1], self.X_r[:, 1:2], self.X_r[:, 2:3], self.X_r[:, 3:4], self.X_r[:, 4:5], self.X_r[:, 5:6])
        loss_r = torch.mean(torch.square(r_pred))
        return loss_r

    def loss_ic(self):
        u_pred = self.net_u(self.X_ic[:, 0:1], self.X_ic[:, 1:2], self.X_ic[:, 2:3], self.X_ic[:, 3:4], self.X_ic[:, 4:5], self.X_ic[:, 5:6])
        loss_ic = torch.mean(torch.square(u_pred.flatten() - self.Y_ic.flatten()))
        return loss_ic

    def loss_ut(self):
        u_t_pred = self.net_ut(self.X_ic[:, 0:1], self.X_ic[:, 1:2], self.X_ic[:, 2:3], self.X_ic[:, 3:4], self.X_ic[:, 4:5], self.X_ic[:, 5:6])
        loss_ut = torch.mean(torch.square(u_t_pred))
        return loss_ut

    def loss_bc(self):
        u_pred = self.net_u(self.x_bc[:,0:1], self.x_bc[:,1:2], self.x_bc[:,2:3], self.x_bc[:,3:4], self.x_bc[:,4:5], self.x_bc[:,5:6])
        loss_bc = torch.mean(torch.square(u_pred.flatten() - self.u_bc.flatten()))
        return loss_bc

    def loss_bc_ux(self):
        u_x_pred = self.net_ux(self.x_bc[:,0:1], self.x_bc[:,1:2], self.x_bc[:,2:3], self.x_bc[:,3:4], self.x_bc[:,4:5], self.x_bc[:,5:6])
        loss_bc_ux = torch.mean(torch.square(u_x_pred.flatten() - self.u_bc.flatten()))
        return loss_bc_ux

    def loss_bc_uxx(self):
        u_xx_pred = self.net_uxx(self.x_bc[:,0:1], self.x_bc[:,1:2], self.x_bc[:,2:3], self.x_bc[:,3:4], self.x_bc[:,4:5], self.x_bc[:,5:6])
        loss_bc_uxx = torch.mean(torch.square(u_xx_pred.flatten() - self.u_bc.flatten()))
        return loss_bc_uxx

    def loss(self, iter):
        L0 = self.loss_ic()
        Lbc = self.loss_bc()
        Lbc_ux = self.loss_bc_ux()
        Lbc_uxx = self.loss_bc_uxx()

        Lr = self.loss_res()
        # L_t = self.lt_loss(collocations)
        # L_t, W = self.residuals_and_weights(self.tol, xx, yy)
        #
        # if W.min().item() > 0.995:
        #     self.S = True
        # # if iter  > 0 and iter % 2000 == 0:
        # #     print(W)
        # if iter > 0 and iter % 50000 == 0:
        #     fig = plt.figure(figsize=(6, 5))
        #     plt.plot(self.t_r, W.cpu().numpy())
        #     plt.ylim([0, 1.0])
        #     plt.xlabel('$t$')
        #     plt.ylabel('$w(t)$')
        #     # plt.show()
        #     plt.savefig(fig_save_path + "weight_" + str(iter))
        #     plt.close()
        #
        #     fig = plt.figure(figsize=(6, 5))
        #     plt.plot(self.t_r, L_t.detach().cpu().numpy())
        #     plt.xlabel('$t$')
        #     plt.ylabel('$\mathcal{L}(t, \\theta)$')
        #     # plt.show()
        #     plt.savefig(fig_save_path + "Lt_" + str(iter))
        #     plt.close()
        # loss = torch.mean(W * L_t) + L0
        loss = L0 + Lr + Lbc + Lbc_ux + Lbc_uxx
        return loss

    # def residuals_and_weights(self,tol, xx,yy):
    #     r_pred = self.net_f(xx,yy)
    #     r_pred = r_pred.reshape(self.nt,self.nx)
    #     L_t = torch.mean(torch.square(r_pred),dim = 1)
    #     W = torch.exp(-tol*(torch.matmul(self.M,L_t))).detach()
    #     return L_t,W

    def train(self,model,model_save_path,pre_model_save_path,nIter = 10000):
        # load previous model
        # model.NN, optimizer, scheduler = load_checkpoint(model.NN, self.optimizer, self.scheduler, pre_model_save_path)

        for param_group in self.optimizer.param_groups:
            print(param_group['lr'])

        pbar = trange(nIter)
        train_loss = []
        window = 1

        # xy1 = lhs(n_x)
        # x_save = torch.Tensor(xy1[:, 0:1])
        # y_save = torch.Tensor(xy1[:, 1:2] * 1)
        # # yy_list = []
        # # yy_list.append(yy1)
        # # for i in range(1, n_t, 1):
        # #     yy = torch.Tensor(xy1[:, 1:2] * 0.01 + 0.01 * i)
        # #     # yy.requires_grad=True
        # #     yy_list.append(yy)
        # #
        # # # list???tensor
        # # # yy_tensor = torch.cat(yy_list,0)
        # # for i in range(n_t):
        # #     y_col = yy_list[i]
        # #     x_col = xx1
        # #     if i == 0:
        # #         y_save = y_col
        # #         x_save = xx1
        # #     else:
        # #         y_save = torch.cat((y_save, y_col), dim=0)
        # #         x_save = torch.cat((x_save, x_col), dim=0)
        #
        # x_save.requires_grad = True
        # y_save.requires_grad = True
        # xy = torch.rand((40000, 2)).to(device)
        # x_save = xy[:,0:1]*5
        # y_save = xy[:,1:2]*20
        # x_save.requires_grad = True
        # y_save.requires_grad = True
        # # x_test.detach()
        # # y_test.detach()
        # # L_t_test.detach()
        # # W_test.detach()
        # # ????
        # fig = plt.figure()
        # ax1 = fig.add_subplot(1,1,1)
        # ax1.set_title('collocation points')
        # ax1.set_xlabel('x')
        # ax1.set_ylabel('t')
        # x = x_save.detach().cpu().numpy().ravel()
        # y = y_save.detach().cpu().numpy().ravel()
        # ax1.scatter(x, y, s=1)
        # # plt.show()
        # plt.savefig(fig_save_path + "original_lhs.png")
        # plt.close()
        # # plt.show(block=True)



        for it in pbar:
            self.optimizer.zero_grad()
            torch.cuda.empty_cache()
            loss = self.loss(it)
            train_loss.append(loss)
            loss.backward()
            self.optimizer.step()

            if self.S:
                break

            if it>0 and it % tim == 0:  # 1000
                loss_value = self.loss(it).detach().cpu().numpy()
                loss_ics_value = self.loss_ic().detach().cpu().numpy()
                # loss_ut_value = self.loss_ut().detach().cpu().numpy()
                loss_bcs_value = self.loss_bc().detach().cpu().numpy()
                loss_bcs_ux_value = self.loss_bc_ux().detach().cpu().numpy()
                loss_bcs_uxx_value = self.loss_bc_uxx().detach().cpu().numpy()
                loss_res_value = self.loss_res().detach().cpu().numpy()
                # loss_u_value = self.Mses_u_np()
                loss_u_value = self.Mses_u().detach().cpu().numpy()
                # loss_ux_value = self.Mses_ux().detach().cpu().numpy()
                # loss_ut_value = self.Mses_ut().detach().cpu().numpy()
                # loss_u_value_l2 = self.Mses_u_l2().detach().cpu().numpy()
                # loss_ux_value_l2 = self.Mses_ux_l2().detach().cpu().numpy()
                # loss_ut_value_l2 = self.Mses_ut_l2().detach().cpu().numpy()
                # # L_t_save,W_save = self.residuals_and_weights(tol,x_save,y_save)
                # L_t_value = L_t_save.detach().cpu().numpy()
                # W_value = W_save.detach().cpu().numpy()

                self.loss_log.append(loss_value)
                self.loss_ics_log.append(loss_ics_value)
                # self.loss_ut_log.append(loss_ut_value)
                self.loss_bcs_log.append(loss_bcs_value)
                self.loss_bcs_ux_log.append(loss_bcs_ux_value)
                self.loss_bcs_uxx_log.append(loss_bcs_uxx_value)
                self.loss_res_log.append(loss_res_value)
                self.loss_s_u_log.append(loss_u_value)
                # self.loss_s_ux_log.append(loss_ux_value)
                # self.loss_s_ut_log.append(loss_ut_value)
                # self.loss_s_u_l2_log.append(loss_u_value_l2)
                # self.loss_s_ux_l2_log.append(loss_ux_value_l2)
                # self.loss_s_ut_l2_log.append(loss_ut_value_l2)
                # self.W_log.append(W_value)
                # self.L_t_log.append(L_t_value)

                pbar.set_postfix({"Loss":loss_value,
                                  "L_ics":loss_ics_value,
                                  # "Loss_ut": loss_ut_value,
                                  "L_bcs": loss_bcs_value,
                                  "L_bc_ux": loss_bcs_ux_value,
                                  "L_bc_uxx": loss_bcs_uxx_value,
                                  "L_res":loss_res_value})

            if it%5000 == 0:
                self.scheduler.step()
            save_checkpoint(model.NN, self.optimizer, self.scheduler, model_save_path)
        return train_loss

    def predict(self,X):
        x = torch.tensor(X[:, 0:1]).float().to(device)
        t = torch.tensor(X[:, 1:2]).float().to(device)
        a = torch.tensor(X[:, 2:3]).float().to(device)
        b = torch.tensor(X[:, 3:4]).float().to(device)
        c = torch.tensor(X[:, 4:5]).float().to(device)
        d = torch.tensor(X[:, 5:6]).float().to(device)

        self.NN.eval()
        u = self.net_u(x,t,a,b,c,d).detach().cpu().numpy()
        return u


def save_checkpoint(model, optimizer, scheduler, save_dir):
    '''save model and optimizer'''

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
        }, save_dir)

def load_checkpoint(model, optimizer, scheduler, save_dir):
    '''load model and optimizer'''

    checkpoint = torch.load(save_dir)
    model.load_state_dict(checkpoint['model_state_dict'])

    if (not optimizer is None):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    print('Pretrained model loaded!')

    return model, optimizer, scheduler

# In[ ]:


x = np.linspace(-10.0, 10.0, 512)
t = np.linspace(-1.0, 1.0, 100)
x_ic = np.linspace(-10.0, 10.0, 2048)
t_ic = -1.0
# T_ic, X_ic = np.meshgrid(t_ic,x_ic)

alpha = 9 #np.linspace(8.7, 9.3, 3)
beta = 0 #np.linspace(-0.4, 0.4, 3)
gamma = np.linspace(0.9, 1.1, 3)
kappa = np.linspace(0.9, 1.1, 3)

T_ic, X_ic, A_ic, B_ic, C_ic, D_ic = np.meshgrid(t_ic,x_ic,alpha,beta,gamma,kappa)

T,X,A,B,C,D = np.meshgrid(t,x,alpha,beta,gamma,kappa)
T_ref,X_ref,A_ref,B_ref,C_ref,D_ref = np.meshgrid(t,x,alpha,beta,gamma,kappa)

usol = B_ref/C_ref + (A_ref-B_ref) / (C_ref * np.cosh(np.sqrt((A_ref-B_ref)/(12*D_ref)) * (X_ref - (B_ref + (A_ref-B_ref)/3) * T_ref)) ** 2)
usol_ic = B_ic/C_ic + (A_ic-B_ic) / (C_ic * np.cosh(np.sqrt((A_ic-B_ic)/(12*D_ic)) * (X_ic - (B_ic + (A_ic-B_ic)/3) * T_ic)) ** 2)

# T_ic_test, X_ic_test = np.meshgrid(t_ic,x_ic)
# usol_ic_test = beta0/gamma + (alpha0-beta0) / (gamma * np.cosh(np.sqrt((alpha0-beta0)/(12*kappa)) * (X_ic_test - (beta0 + (alpha0-beta0)/3) * T_ic_test)) ** 2)

# usol_ux = -9*np.sqrt(3)*np.tanh(np.sqrt(3/4)*(X_ref - 3*T_ref)) / np.cosh(np.sqrt(3) / 2 * (X_ref - 3 * T_ref)) ** 2
# usol_ut = -3*usol_ux

M = 0
d0 = M*2 + 6
layers = [d0, 80, 80, 1]


n_t = 100 #512
n_x = 512
tol = 100
tim = 1000

state0 = usol_ic #usol[:,0:1]

fig_save_path = 'C:/Users/18090/Desktop/4th-revise/fig/'
pre_model_save_path = 'C:/Users/18090/Desktop/4th-revise/kdv_uq_cd.pt'
model_save_path = 'C:/Users/18090/Desktop/4th-revise/kdv_uq_cd-test.pt'


model = CasualPINN(layers,M,x_ic,state0,usol,alpha,beta,gamma,kappa,n_t,n_x,tol)  # usol_ux,usol_ut,

# train_loss = model.train(model,model_save_path,pre_model_save_path,nIter=20001)  # 300000  , model_save_path, pre_model_save_path
#
# train_loss_tensor = torch.stack(train_loss,dim=0)
# train_loss_numpy = train_loss_tensor.detach().cpu().numpy()

model.NN, _, _ = load_checkpoint(model.NN, model.optimizer, model.scheduler, pre_model_save_path)

# # epoch = range(tim*len(model.loss_ics_log))
# # x_ticks = np.arange(tim, tim*(len(model.loss_ics_log)+1), tim)
# # plt.xticks(x_ticks)
# plt.plot(model.loss_ics_log, label = 'loss_ic')
# plt.plot(model.loss_bcs_log, label = 'loss_bc')
# plt.plot(model.loss_res_log, label = 'loss_res')
# plt.yscale("log")
# # plt.xlabel(f'Iteration$(x10^2)$')
# plt.legend(loc='upper right')
# plt.savefig(fig_save_path + "loss_ics_res.png")
# plt.close()

# np.savetxt(fig_save_path + "loss_ic.txt", model.loss_ics_log)
# np.savetxt(fig_save_path + "loss_bc.txt", model.loss_bcs_log)
# np.savetxt(fig_save_path + "loss_bcx.txt", model.loss_bcs_ux_log)
# np.savetxt(fig_save_path + "loss_bcxx.txt", model.loss_bcs_uxx_log)
# np.savetxt(fig_save_path + "loss_res.txt", model.loss_res_log)
#
# np.savetxt(fig_save_path + 'loss_s_u.txt', model.loss_s_u_log)
# np.savetxt(fig_save_path + 'loss_s_ux.txt', model.loss_s_ux_log)
# np.savetxt(fig_save_path + 'loss_s_ut.txt', model.loss_s_ut_log)
# np.savetxt(fig_save_path + 'loss_s_u_l2.txt', model.loss_s_u_l2_log)
# np.savetxt(fig_save_path + 'loss_s_ux_l2.txt', model.loss_s_ux_l2_log)
# np.savetxt(fig_save_path + 'loss_s_ut_l2.txt', model.loss_s_ut_l2_log)
# In[ ]:

X_p = np.concatenate([X.flatten()[:,None],T.flatten()[:,None],A.flatten()[:,None],B.flatten()[:,None],C.flatten()[:,None],D.flatten()[:,None]],axis = 1)
u_pred = model.predict(X_p).reshape(usol.shape)


# u_err_ab = (usol - u_pred)[:,:,:,:,:,:]
# u_err_mean = np.mean(u_err_ab)
# print('mean is', u_err_mean)
# u_err_var = np.var(u_err_ab)
# print('var is', u_err_var)

error = np.linalg.norm((usol - u_pred)[:,:,:,:,:,:])/np.linalg.norm(usol)
mse = np.mean(np.square((usol - u_pred)[:,:,:,:,:,:]))
print(f"Relative l2 error: {error:.3e}")
print(f"mse: {mse:.3e}")

# X_p_ic = np.concatenate([X_ic.flatten()[:,None],T_ic.flatten()[:,None],A_ic.flatten()[:,None],B_ic.flatten()[:,None],C_ic.flatten()[:,None],D_ic.flatten()[:,None]],axis = 1)
# u_pred_ic = model.predict(X_p_ic).reshape(usol_ic.shape)
# u_err_ab_ic = (usol_ic - u_pred_ic)[:,:,:,:,:,:]
# u_err_mean_ic = np.mean(u_err_ab_ic)
# u_err_var_ic = np.var(u_err_ab_ic)
# print('mean_ic is', u_err_mean_ic)
# print('var_ic is', u_err_var_ic)

#### initial snapshot-uq result
mean_u0_ab = np.mean(u_pred[:,0:1,:,:,:,:], axis=(4,5), keepdims=True)
mean_u0_ab_result = mean_u0_ab.reshape(-1,1)
mean_u0_ab_exact = np.mean(usol[:,0:1,:,:,:,:], axis=(4,5))
mean_u0_ab_exact_result = mean_u0_ab_exact.reshape(-1,1)
var_u0_ab = np.var(u_pred[:,0:1,:,:,:,:], axis=(4,5))
var_u0_ab_result = var_u0_ab.reshape(-1,1)

diff_square = (u_pred[:,0:1,:,:,:,:] - mean_u0_ab) ** 2
var_u_pred = np.mean(diff_square, axis=(4,5))


# np.savetxt(fig_save_path + "usol_ic.txt", usol[:,0:1,:,:,:,:].reshape(-1,1))
# np.savetxt(fig_save_path + "upred_ic.txt", u_pred[:,0:1,:,:,:,:].reshape(-1,1))
# np.savetxt(fig_save_path + "umean_ic.txt", mean_u0.reshape(-1,1))
# np.savetxt(fig_save_path + "uvar_ic.txt", var_u0.reshape(-1,1))

plt.plot(x, mean_u0_ab_exact_result, color='blue',label='exact,t=-1')
plt.plot(x, mean_u0_ab_result, '--', color='red',label='predict,t=-1')
plt.fill_between(x, mean_u0_ab_result.reshape(-1) - var_u_pred.reshape(-1), mean_u0_ab_result.reshape(-1) + var_u_pred.reshape(-1), color='tomato', alpha=0.4)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('$x$',fontsize=20,labelpad=1)
plt.ylabel('$u$',fontsize=20,labelpad=1)
plt.legend(fontsize=14)
plt.savefig(fig_save_path + 'ic_mean_std_cd.pdf', dpi = 300)
plt.close()

#### end snapshot-uq result
mean_ue_ab = np.mean(u_pred[:,99:100,:,:,:,:], axis=(4,5), keepdims=True)
mean_ue_ab_result = mean_ue_ab.reshape(-1,1)
mean_ue_ab_exact = np.mean(usol[:,99:100,:,:,:,:], axis=(4,5))
mean_ue_ab_exact_result = mean_ue_ab_exact.reshape(-1,1)
var_ue_ab = np.var(u_pred[:,99:100,:,:,:,:], axis=(4,5))
var_ue_ab_result = var_ue_ab.reshape(-1,1)

diff_square_e = (u_pred[:,99:100,:,:,:,:] - mean_ue_ab) ** 2
var_u_pred_e = np.mean(diff_square_e, axis=(4,5))

plt.plot(x, mean_ue_ab_exact_result, color='blue',label='exact,t=1')
plt.plot(x, mean_ue_ab_result, '--', color='red',label='predict,t=1')
plt.fill_between(x, mean_ue_ab_result.reshape(-1) - var_u_pred_e.reshape(-1), mean_ue_ab_result.reshape(-1) + var_u_pred_e.reshape(-1), color='tomato', alpha=0.4)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('$x$',fontsize=20,labelpad=1)
plt.ylabel('$u$',fontsize=20,labelpad=1)
plt.legend(fontsize=14)
plt.savefig(fig_save_path + 'ic_mean_std_cd_e.pdf', dpi = 300)
plt.close()

ag =100
# mean_u_a = np.mean(usol[:,0:1,0:1,:,:,:] - u_pred[:,0:1,0:1,:,:,:], axis=(1,2,3,4,5), keepdims=True)
# var_u_a = np.var(usol[:,0:1,0:1,:,:,:] - u_pred[:,0:1,0:1,:,:,:], axis=(1,2,3,4,5), keepdims=True)
#
# mean_u_a_result = np.abs(mean_u_a).reshape(-1,1)
# var_u_a_result = np.abs(var_u_a).reshape(-1,1)

# np.savetxt(fig_save_path + "u_err_mean_ic.txt", u_err_mean_ic)
# np.savetxt(fig_save_path + "u_err_var_ic.txt", u_err_var_ic)
# np.savetxt(fig_save_path + "u_err_ab_ic.txt", u_err_ab_ic)

# plt.figure()
# plt.plot(train_loss_numpy, label = 'train loss')
# plt.yscale('log')
# # plt.xlabel(f'Iteration$(x10^2)$')
# plt.legend()
# plt.savefig(fig_save_path + 'train loss.png', dpi = 300)
# plt.close()
# np.savetxt('kdv_our_loss.txt',train_loss_numpy)
# In[ ]:

# fig = plt.figure(figsize=(18,4))
# ax = fig.add_subplot(1,3,1)
# plt.pcolormesh(T,X,usol,cmap="jet",shading='gouraud')
# plt.colorbar()
# plt.xlabel("$t$",fontsize=20,labelpad=1)
# plt.ylabel("$x$",fontsize=20,labelpad=1)
# plt.title("Exact $u(x,t)$")
#
# ax = fig.add_subplot(1,3,2)
# plt.pcolormesh(T,X,u_pred,cmap="jet",shading='gouraud')
# plt.colorbar()
# plt.xlabel("$t$",fontsize=20,labelpad=1)
# plt.ylabel("$x$",fontsize=20,labelpad=1)
# plt.title("Predicted $u(x,t)$")
#
# ax = fig.add_subplot(1,3,3)
# plt.pcolormesh(T,X,np.abs(usol - u_pred),cmap="jet",shading='gouraud')
# plt.colorbar()
# plt.xlabel("$t$",fontsize=20,labelpad=1)
# plt.ylabel("$x$",fontsize=20,labelpad=1)
# plt.title("Absolute Error")
# plt.savefig(fig_save_path + 'bigAbsolute Error.png', dpi = 300)
# plt.close()


# fig = plt.figure(figsize=(13, 4))
# plt.subplot(1, 3, 1)
# plt.plot(x, usol_test[:,0], color='blue',label='exact')
# plt.plot(x, upred_test[:,0], '--', color='red',label='predict')
# plt.xlabel('$x$',fontsize=20,labelpad=1)
# plt.ylabel('$u$',fontsize=20,labelpad=1)
# plt.xticks(fontsize=8)
# plt.yticks(fontsize=8)
# plt.title('$t = -1.0$',fontsize=20)
# plt.legend()
# plt.tight_layout()
#
# plt.subplot(1, 3, 2)
# plt.plot(x, usol_test[:,50], color='blue',label='exact')
# plt.plot(x, upred_test[:,50], '--', color='red',label='predict')
# plt.xlabel('$x$',fontsize=20,labelpad=1)
# plt.ylabel('$u$',fontsize=20,labelpad=1)
# plt.xticks(fontsize=8)
# plt.yticks(fontsize=8)
# plt.title('$t = 0$',fontsize=20)
# plt.legend()
# plt.tight_layout()
#
# plt.subplot(1, 3, 3)
# plt.plot(x, usol_test[:,-1], color='blue',label='exact')
# plt.plot(x, upred_test[:,-1], '--', color='red',label='predict')
# plt.xlabel('$x$',fontsize=20,labelpad=1)
# plt.ylabel('$u$',fontsize=20,labelpad=1)
# plt.xticks(fontsize=8)
# plt.yticks(fontsize=8)
# plt.title('$t = 1.0$',fontsize=20)
# plt.legend()
# plt.tight_layout()
# plt.savefig(fig_save_path + 'bigfinal_result_legend.png', dpi = 300)
# plt.close()
#
# np.savetxt(fig_save_path + "T.txt", T)
# np.savetxt(fig_save_path + "X.txt", X)
# np.savetxt(fig_save_path + "u_pred.txt", u_pred)
# np.savetxt(fig_save_path + "usol.txt", usol)
# np.savetxt(fig_save_path + "uerror.txt", np.abs(usol - u_pred))
# np.savetxt(fig_save_path + "snap_x.txt", x)
# np.savetxt(fig_save_path + "snap_usol0.txt", usol[:,0])
# np.savetxt(fig_save_path + "snap_upred0.txt", u_pred[:,0])
# np.savetxt(fig_save_path + "snap_usol0.5.txt", usol[:,50])
# np.savetxt(fig_save_path + "snap_upred0.5.txt", u_pred[:,50])
# np.savetxt(fig_save_path + "snap_usol1.txt", usol[:,-1])
# np.savetxt(fig_save_path + "snap_upred1.txt", u_pred[:,-1])
