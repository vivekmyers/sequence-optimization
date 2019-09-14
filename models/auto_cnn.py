import models.autoencoder

def CNN(*args, **kwargs):
    '''Use autoencoder architecture for predictive model.'''
    model = models.autoencoder.Autoencoder(*args, **kwargs, dim=100, beta=1.)
    model.fit = model.fit
    return model
