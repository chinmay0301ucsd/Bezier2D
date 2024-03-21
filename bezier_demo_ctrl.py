## Fit Bezier Curve to Heart Shaped Equation
import torch
import slangpy
import os 
import matplotlib.pyplot as plt

N = 20
c = 2 
# m = slangpy.loadModule('bezier_compute.slang', defines={"NUM_CTRL_PTS": N, "DIM":c})
m = slangpy.loadModule('bezier.slang', defines={"NUM_CTRL_PTS": N, "DIM":c})


def plot_sdf(num_pts, control_pts, savedir):
	"""
	num_pts : int - compute sdf for num_pts x num_pts points
	control_pts : torch tensor (N,2) on GPU
	"""

	xmin, xmax = torch.min(control_pts[:,0]), torch.max(control_pts[:,0])
	ymin, ymax = torch.min(control_pts[:,1]), torch.max(control_pts[:,1])
	px = torch.linspace(xmin.item()-2, xmax.item()+2, num_pts)
	py = torch.linspace(ymin.item()-2, ymax.item()+2, num_pts)
	print(xmin, xmax, ymin, ymax)
	# Create the meshgrid
	x, y = torch.meshgrid(px, py.flip(dims=[0]), indexing='ij')  # 'i
	xy = torch.stack([x,y], dim=-1 ).view(-1,2).cuda()
	sdf_mats = torch.zeros(xy.shape[0], c*(N-1), c*(N-1)).cuda()

	# m.bezier2DSDFtest(xy=xy, control_pts=control_pts, output=sdf).launchRaw(blockSize=(1024, 1, 1), gridSize=(1024, 1, 1))
	m.bezier2DSDF(xy=xy, control_pts=control_pts, output=sdf_mats).launchRaw(blockSize=(256, 1, 1), gridSize=(64, 1, 1))
	sdf = torch.linalg.det(sdf_mats)
	sdf = torch.sgn(sdf) * torch.sqrt(torch.abs(sdf))

	sdf_plot = sdf.view(num_pts, num_pts).cpu().numpy()
	plt.figure()
	plt.imshow(sdf_plot.T, cmap='inferno')
	plt.title(f'Implicitized SDF of Bezier Curve with {N} Control Points')
	plt.colorbar()
	plt.savefig(os.path.join(savedir, f'Bcurve_{N}_SDF.png'))

def heart(t):
    t = t*2*torch.pi
    x = 16*(torch.sin(t))**3
    y = 13*torch.cos(t) - 5*torch.cos(2*t) -2*torch.cos(3*t) - torch.cos(4*t)
    return torch.hstack([x.reshape(-1,1),y.reshape(-1,1)])

def ellipse(t, a, b):
    t = t*2*torch.pi
    x = a * (torch.cos(t))
    y = b * (torch.sin(t))
    return torch.hstack([x.reshape(-1,1),y.reshape(-1,1)])

def astrid(t, a):
    t = t*2*torch.pi
    x = a * (torch.cos(t))**3
    y = a * (torch.sin(t))**3
    return torch.hstack([x.reshape(-1,1),y.reshape(-1,1)])

class Bezier2D(torch.autograd.Function):
	@staticmethod
	def forward(ctx, t, control_pts):
		"""
		t: M,1 (torch.tensor) on GPU, parameter for bezier curves
		control_pts: N,2 (torch.tensor) 
		"""
		# coeffs = torch.zeros_like(control_pts, dtype=torch.float).cuda()
		# m.compute_coeffs(control_pts=control_pts, output=coeffs).launchRaw(blockSize=(1, 1, 1), gridSize=(1, 1, 1))
		outputs = torch.zeros(t.shape[0], control_pts.shape[1]).cuda()
		kernel_with_args = m.bezier2D(t=t, control_pts=control_pts, output=outputs)
		NUM_BLOCKS = 1 + t.shape[0] // 1024
		kernel_with_args.launchRaw(
			blockSize=(NUM_BLOCKS, 1, 1),
			gridSize=(1024, 1, 1))
		ctx.save_for_backward(t, control_pts, outputs)
		return outputs

	@staticmethod
	def backward(ctx, grad_outputs):
		(t, control_pts, outputs) = ctx.saved_tensors
		grad_ctrl_pts = torch.zeros_like(control_pts).cuda()
		grad_t  = torch.zeros_like(t).cuda()
  		# Note: When using DiffTensorView, grad_output gets 'consumed' during the reverse-mode.
		# If grad_output may be reused, consider calling grad_output = grad_output.clone()

		kernel_with_args = m.bezier2D.bwd(t=(t, grad_t),
                                                       control_pts=(control_pts, grad_ctrl_pts),
                                                       output=(outputs, grad_outputs))
		NUM_BLOCKS = 1 + t.shape[0] // 1024
		kernel_with_args.launchRaw(
			blockSize=(NUM_BLOCKS, 1, 1),
			gridSize=(1024, 1, 1))

		return grad_t, grad_ctrl_pts



def curve_from_coeffs(t, coeffs):
    """ To check if coefficients are correct """
    output = torch.zeros(t.shape[0], coeffs.shape[1]).cuda()
    for i in range(coeffs.shape[0]):
        output = output + (t**i).view(-1,1) * coeffs[i].view(1,-1)
    return output 

num_pts = 100
t = torch.linspace(0.0, 1, num_pts, dtype=torch.float).cuda()

savedir =  "./astrid_20"
os.makedirs(savedir, exist_ok=True)
gt_pts = ellipse(t, 3.0, 4.0).cuda() #heart(t).cuda()
gt_pts = astrid(t, 3.0)
# gt_pts = heart(t).cuda()
control_pts = 1*torch.rand((N, 2), dtype=torch.float).cuda()
# control_pts = torch.cat((control_pts, control_pts[0].unsqueeze(0)), dim=0)
control_pts.requires_grad_(True)


### Experiment 1 - Learning control points to match heart
# Define a custom parameter, for example, a single value parameter.
opt_param = torch.nn.Parameter(control_pts)
import matplotlib.pyplot as plt 
plt.figure()
pts = Bezier2D.apply(t,  opt_param)
plt.plot(pts[:,0].detach().cpu().numpy()/0.9, pts[:,1].detach().cpu().numpy()/0.9, color='red',label='predicted')
plt.savefig(os.path.join(savedir, 'init.png'))

# Use an optimizer, for example, SGD, and register the custom parameter with it.
lr_init = 0.01
optimizer = torch.optim.Adam([opt_param], lr=lr_init)

loss_curve = []
# Example training loop
for epoch in range(10000):  # Assuming 100 epochs
    pts = Bezier2D.apply(t,  opt_param)
    loss = ((torch.linalg.norm(pts  - gt_pts, dim=1))).mean()
    # Zero gradients (necessary to clear previous gradients)
    optimizer.zero_grad()
    for pg in optimizer.param_groups:
        pg['lr'] = lr_init * 0.99
    # Backpropagation
    loss.backward()
    # Update the parameter based on the current gradients
    optimizer.step()
    loss_curve.append(loss.item())
    # Print the parameter's value and loss if you want to monitor the progress
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

import matplotlib.pyplot as plt 
plt.figure()
pts = Bezier2D.apply(t,  opt_param)
plt.plot(pts[:,0].detach().cpu().numpy()/0.95, pts[:,1].detach().cpu().numpy()/0.95, color='red',label='predicted')
plt.plot(gt_pts[:,0].detach().cpu().numpy(), gt_pts[:,1].detach().cpu().numpy(), color='green',label='gt')
plt.legend(['Predicted', 'Ground Truth'])
plt.savefig(os.path.join(savedir, 'control_pts_descent.png'))
plt.figure()
plt.plot(loss_curve)
plt.title('Loss Curve')
plt.xlabel('Iterations')
plt.ylabel('Loss Value')
plt.savefig(os.path.join(savedir, 'Loss_Curve.png'))

plot_sdf(100, opt_param, savedir)

# plt.scatter(0.5, 0.6)
# plt.text(0.52, 0.6, 'Initialization')
# plt.title(f'Bezier Curve with {N} Control Points')
# breakpoint()
# gt_sdf = compute_sdf(gt_control_pts, 1, [0.0, 1.0], [0.0, 1.0])

# sdf = compute_sdf(control_pts, 1, [0.5, 0.6], [0.5, 0.6])
# loss = sdf.backward()
# print(control_pts.grad)
# print(xy.grad)
# output = torch.zeros((num_pts,2), dtype=torch.float).cuda()

# # Number of threads launched = blockSize * gridSize
# m.bezier2D(t=t, control_pts=control_pts, output=output).launchRaw(blockSize=(32, 1, 1), gridSize=(64, 1, 1))

