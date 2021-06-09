import torch


def compute_pairwise_distance(x, y):
    r''' Computes the squared pairwise Euclidean distances between x and y

    Args:
        x: a tensor of shape [num_x_samples, num_features]
        y: a tensor of shape [num_y_samples, num_features]

    Returns:
        a distance matrix of dimensions [num_x_samples, num_y_samples]

    Raise:
        ValueError: if the inputs do no matched the specified dimensions.
    '''

    if not len(x.size()) == len(y.size()) == 2:
        raise ValueError('Both inputs should be matrices')
    if x.size()[1] != y.size()[1]:
        raise ValueError('The number of features should be the same.')

    # By making the `inner` dimensions of the two matrices equal to 1 using
    # broadcasting then we are essentially substracting every pair of rows
    # of x and y.
    def norm(x): return torch.sum(x * x, 1)
    return norm(x.unsqueeze(2) - y.t())


def gaussian_kernel_matrix(x, y, sigmas):
    r''' Computes a Gaussian RBK between the samples of x and y.

    We create a sum of multiple gaussian kernels each having a width signa_i.

    Args:
        x: a tensor of shape [num_samples, num_features]
        y: a tensor of shape [num_samples, num_features]
        sigmas: a tensor of floats which denote the widths of each of the
                gaussians in the kernel
    Returns:
        A tensor of shape [num_samples{x}, num_samples{y}] with the RBF kernel
    '''
    beta = 1. / (2. * (sigmas.unsqueeze(1)))

    dist = compute_pairwise_distance(x, y)

    s = torch.matmul(beta, dist.view(1, -1))

    return torch.sum(torch.exp(-s), 0).view_as(dist)


def maximum_mean_discrepancy(x, y, kernel=gaussian_kernel_matrix):
    r'''Computes the Maximul Mean Discrepancy (MMD) of two samples: x and y.

    Maximum Mean Discrepancy (MMD) is a distance-measure between the samples of
    the distributions of x and y. Here we use kernel tow sample estimate
    using the empirical mean of the two distributions.

    Args:
        x: a tensor of shape [num_samples, num_features]
        y: a tensor of shape [num_samples, num_features]
        kernel: a function which computes the kernel in MMD. Defaults to the
        FaussianKernelMatrix

    Returns:
        a scalar denoting the squared maximum mean discrepancy loss
    '''

    cost = torch.mean(kernel(x, x))
    cost += torch.mean(kernel(y, y))
    cost -= 2 * torch.mean(kernel(x, y))

    cost = torch.clamp(cost, min=0)
    return cost


def mahalanobis_metric_fast(p, mu, U, pos_mu, pos_U, neg_mu, neg_U):
    # covi = (cov + I).inverse()
    mahalanobis_distances = (p - mu).mm(U.mm(U.t())).mm((p - mu).t())
    pos_mahalanobis_distance = (
        p - pos_mu).mm(pos_U.mm(pos_U.t())).mm((p - pos_mu).t()).diag().sqrt().data
    neg_mahalanobis_distance = (
        p - neg_mu).mm(neg_U.mm(neg_U.t())).mm((p - neg_mu).t()).diag().sqrt().data
    mahalanobis_ratio1 = pos_mahalanobis_distance - neg_mahalanobis_distance
    mahalanobis_ratio2 = neg_mahalanobis_distance - pos_mahalanobis_distance
    max_ratio = torch.max(mahalanobis_ratio1, mahalanobis_ratio2)

    return max_ratio  # / TEMPERATURE
    # return mahalanobis_distances.diag().sqrt().data


def mahalanobis_metric(p, S, L, U, pos_U, neg_U, args, encoder=None):
    r''' Compute the mahalanobis distance between the encoding of a sample (p) and a set (S).

    Args:
        p: tensor (batch_size, dim), a batch of samples
        S: tensor (size, dim), a domain which contains a set of samples
        encoder: a module used for encoding p and S

    Return:
        mahalanobis_distances: tensor (batch_size)
    '''

    if encoder is not None:
        p = encoder(p)  # (batch_size, dim)
        S = encoder(S)  # (size, dim)

    neg_index = ((L == 0).nonzero())
    pos_index = ((L == 1).nonzero())

    neg_index = neg_index.expand(neg_index.size(0), S.data.size(1))
    pos_index = pos_index.expand(pos_index.size(0), S.data.size(1))

    neg_S = torch.gather(S, 0, neg_index)
    pos_S = torch.gather(S, 0, pos_index)
    neg_mu = torch.mean(neg_S, dim=0, keepdim=True)
    pos_mu = torch.mean(pos_S, dim=0, keepdim=True)

    pos_mahalanobis_distance = (
        p - pos_mu).mm(pos_U.mm(pos_U.t())).mm((p - pos_mu).t()).diag().sqrt()
    neg_mahalanobis_distance = (
        p - neg_mu).mm(neg_U.mm(neg_U.t())).mm((p - neg_mu).t()).diag().sqrt()

    mahalanobis_ratio1 = pos_mahalanobis_distance - neg_mahalanobis_distance
    mahalanobis_ratio2 = neg_mahalanobis_distance - pos_mahalanobis_distance

    max_ratio = torch.max(mahalanobis_ratio1, mahalanobis_ratio2)

    return max_ratio.clamp(0.01, 2)  # / TEMPERATURE # .clamp(0.001, 1)

    # mu_S = torch.mean(S, dim=0, keepdim=True) # (1, dim)
    # mahalanobis_distances = (p - mu_S).mm(U.mm(U.t())).mm((p - mu_S).t())
    # return mahalanobis_distances.diag().sqrt().clamp(0.01, 2)


def biaffine_metric_fast(p, mu, U):
    biaffine_distances = p.mm(U).mm(mu.t())
    return biaffine_distances.squeeze(1).data


def biaffine_metric(p, S, U, W, V, args, encoder=None):
    ''' Compute the biaffine distance between the encoding of a sample (p) and a set (S).

    Args:
        p: tensor (batch_size, dim), a batch of samples
        U: matrix (dim, dim)
        S: tensor (size, dim), a domain which contains a set of samples
        encoder: a module used for encoding p and S

    Return:
        biaffine_distance: tensor (batch_size)
    '''

    if encoder is not None:
        p = encoder(p)
        S = encoder(S)

    mu_S = torch.mean(S, dim=0, keepdim=True)
    biaffine_distances = p.mm(U).mm(mu_S.t()) + \
        p.mm(W) + mu_S.mm(V)  # extra components
    return biaffine_distances.squeeze(1).clamp(-10, 10)
