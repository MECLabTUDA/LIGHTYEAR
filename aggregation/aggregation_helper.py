import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.utils.data import Dataset, DataLoader



class Representor:
    def __init__(self, input_dim, latent_dim):
        self.cvae = CVAE(input_dim = input_dim, condition_dim = 1, latent_dim = latent_dim)
        self.lr = 1e-3
        self.optimizer = torch.optim.Adam(self.cvae.parameters(), lr=self.lr)
        self.threshold = 0

    def compute_loss(self, x, recon_x, mu, logvar):
        #Reconstruction loss
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        #Kl Divergence
        kl_loss    = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_loss, recon_loss, kl_loss

    def train(self, updates, train_round):
        self.cvae.train()
        self.cvae.to('cuda:4')
        for update in updates:
            update = torch.Tensor(update.state_dict_oneD).to('cuda:4')
            self.optimizer.zero_grad()
            recon_update, mu, logvar, z = self.cvae(update, train_round)
            loss, recon_loss, kl_loss = self.compute_loss(update, recon_update, mu, logvar)
            loss.backward()
            self.optimizer.step()

    def _pseudo_labels(self, updates, threshold=None, clustering=False):
        # class 0 -> benign
        # class 1 -> malf
        if not clustering:
            for update in updates:
                if update.meta['recon'] <= threshold:
                    update.meta['pseudo_label'] = 0
                else:
                    update.meta['pseudo_label'] = 1
                update.meta['confidence'] =  abs(update.meta['recon'] - threshold)
            return updates
        
        else:
            recon_errs     = [update.meta['recon'] for update in updates.reshape(-1, 1)]
            kmeans         = KMeans(n_clusters=2, random_state=42, n_init=10)
            kmeans.fit(recon_errs)
            sorted_centers = np.argsort(kmeans.cluster_centers_.flatten())
            mapping        = {sorted_centers[0]: 0, sorted_centers[1]: 1}
            labels         = [mapping[label] for label in kmeans.labels_]

            for update, label in zip(updates, labels):
                update.meta['pseudo_labels'] = label
            return updates

    def _fit(self, updates, train_round):
        
        self.cvae.eval()
        self.cvae.to('cuda:4')
        recons = []

        with torch.no_grad():
            for update in updates:
                recon_update, mu, logvar, z = self.cvae(torch.Tensor(update.state_dict_oneD).to('cuda:4'), train_round)
                update.meta['latent'] = z.cpu().numpy()
                _, update.meta['recon'], _  = self.compute_loss(update.state_dict_oneD, recon_update, mu, logvar)
                recons.append(update.meta['recon'])
            threshold = np.mean(recons)
        return updates, threshold

    def get_representation(self, updates, train_round):
        updates, threshold = self._fit(updates, train_round)
        updates            = self._pseudo_labels(update, threshold)
        return updates
        

    
# Think about pertubations
# Penalize for reconstructing pure noise (random updates)
# Use feature loss to ensure that the updates are remain untouched 
class CVAE(nn.Module):
    
    def __init__(self, input_dim, condition_dim, latent_dim):
        super(CVAE, self).__init__()
        
        self.input_dim     = input_dim
        self.condition_dim = condition_dim
        self.latent_dim    = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim + condition_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
            )

        self.mu     = nn.Linear(64, latent_dim)
        self.logvar = nn.Linear(64, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, 64), 
            nn.ReLU(),
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
            )

    def encode(self, x, c):
        c = torch.tensor([c], dtype=x.dtype, device=x.device)
        c = c.unsqueeze(0).expand(x.size(0), -1)
        h      = self.encoder(torch.cat([x, c.view(1, 1, 1)], dim=-1))
        mu     = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        c = torch.tensor([c], dtype=x.dtype, device=x.device)
        c = c.unsqueeze(0).expand(x.size(0), -1)
        return self.decoder(torch.cat([z, c.view(1, 1, 1)], dim=-1))

    def forward(self, x, c):
        # Encode
        mu, logvar   = self.encode(x, c)
        z            = self.reparameterize(mu, logvar)
        print(z.size())
        print(z)
        recon_update = self.decoder(z, c)
        return recon_update, mu, logvar, z 


class BinaryClassificationNetwork(nn.Module):
    def __init__(self, input_dim):
        super(BinaryClassificationNetwork, self).__init__()
        self.network = nn.Sequential( 
            #nn.Linear(input_dim, 256),
            #nn.ReLU(),
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Single output for binary classification
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.network(x)
        return self.sigmoid(x)


class WeightedBCELoss(nn.Module):
    def __init__(self):
        super(WeightedBCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, outputs, targets, confidence_scores):
        print('Insidse loss function')
        print('outputs')
        print(outputs)
        print('targets')
        print(targets)
        print('confidence score')
        print(confidence_scores)
        loss = self.bce(outputs, targets)
        loss = loss
        weighted_loss = (loss * confidence_scores.item())
        return weighted_loss


class Classifier(nn.Module):
    def __init__(self, input_dim, epochs=5):
        super(Classifier, self).__init__()
        self.model   = BinaryClassificationNetwork(input_dim)
        self.loss_fn = WeightedBCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, updates, epochs=5):
        self.model.train()
        self.model.to('cuda:4')
        for epoch in range(epochs):

            for update in updates:
                vec        = torch.Tensor(update.state_dict_oneD).to('cuda:4')
                label      = torch.Tensor([update.meta['pseudo_label']]).view(1, 1).to('cuda:4')
                confidence = update.meta['confidence']
                pred       = self.model(vec)
                loss       = self.loss_fn(pred, label, confidence)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                

    def final_decision(self, updates):
        self.model.eval()
        self.model.to('cuda:4')
        with torch.no_grad():
            for update in updates:
                vec  = torch.Tensor(update.state_dict_oneD).to('cuda:4')
                pred = self.model(vec) == 0
                
                update.meta['mal'] =  pred
       
        return updates


class NoiseReductionVAE(nn.Module):
    def __init__(self, input_dim):
        super(NoiseReductionVAE, self).__init__()

        latent_dim = 32

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar 


class Regularizor(nn.Module):
    
    def __init__(self):
        super(Regularizor, self).__init__()
        self.unet = UNet1D()
        self.optimizer = torch.optim.Adam(self.unet.parameters(), lr=1e-3)
        self.alpha_space = []
        


    def loss_fn(self, pred, x):
        loss = nn.MSELoss()(pred, x)
        return loss

    def train(self, updates, epochs=5):
        self._gen_alpha_space(updates)


        self.unet.train()
        self.unet.to('cuda:4')

        for epoch in range(epochs):
            for update in updates:
                self.optimizer.zero_grad()
                
                lc_weights    = torch.rand(len(self.alpha_space))
                lc_weights   /= lc_weights.sum()
                alpha_shift   = sum(w * t for w, t in zip(lc_weights, self.alpha_space))
                shifted_input = torch.Tensor(update.state_dict_oneD) + alpha_shift
                shifted_input = torch.Tensor(shifted_input).unsqueeze(1).to('cuda:4')
                reg_vector    = self.unet(shifted_input)
                
                loss = self.loss_fn(reg_vector, torch.Tensor(update.state_dict_oneD).unsqueeze(1).to('cuda:4'))

                loss.backward()
                self.optimizer.step()


    def _gen_alpha_space(self, updates):
        for i in range(len(updates)):
            for j in range(len(updates)):
                if i != j:
                    noise_ij = updates[i].state_dict_oneD - updates[j].state_dict_oneD
                    self.alpha_space.append(torch.Tensor(noise_ij))


    def regularize_updates(self, updates):
        self.unet.eval()
        self.unet.to('cuda:4')
        for update in updates:
            if update.meta['mal']:
                _update = torch.Tensor(update.state_dict_oneD).to('cuda:4')
                regularized_update = self.unet(_update)
                regularized_update       = regularized_update.cpu().numpy()
                update.state_dict_oneD   = regularized_update
                update.oneD_to_state_dict(regularized_update)
        return updates


class Unet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Unet, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(input_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)

        # Decoder
        self.dec4 = self.upconv_block(1024, 512)
        self.dec3 = self.upconv_block(512, 256)
        self.dec2 = self.upconv_block(256, 128)
        self.dec1 = self.upconv_block(128, 64)

        # Final output layer
        self.final_conv = nn.Conv2d(64, output_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            self.conv_block(out_channels, out_channels)
        )

    def forward(self, x):
        # Encoding path
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, kernel_size=2))
        enc3 = self.enc3(F.max_pool2d(enc2, kernel_size=2))
        enc4 = self.enc4(F.max_pool2d(enc3, kernel_size=2))

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, kernel_size=2))

        # Decoding path
        dec4 = self.dec4(bottleneck) + enc4
        dec3 = self.dec3(dec4) + enc3
        dec2 = self.dec2(dec3) + enc2
        dec1 = self.dec1(dec2) + enc1

        # Final output
        return self.final_conv(dec1)


class RegularizorUnet(nn.Module):
 
    def __init__(self, input_dim):
        super(RegularizorUnet, self).__init__()
        self.unet = Unet(input_dim)
        self.optimizer = torch.optim.Adam(self.vae.parameters(), lr=1e-3)

    def loss_fn(self, recon_x, x, mu, logvar):
        recon_loss = nn.MSELoss()(recon_x, x)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_loss
  
    def train(self, updates, epochs=5):
        alpha_vectors = self._gen_alpha_space(updates)
        alpha_space   = torch.stack(alpha_vectors)

        self.vae.train()
        self.vae.to('cuda:4')

        for epoch in epochs:
            for update in updates:
                self.optimizer.zero_grad()

                coeffs      = torch.randn(1, len(alpha_vectors))
                noise       = torch.mm(coeffs, alpha_space)
                noisy_input = update.state_dict_oneD + noise
                noisy_input = torch.Tensor(noisy_input).to('cuda:4')

                reconstructed, mu, logvar = self.vae(noisy_input)
                
                loss = self.loss_fn(reconstructed, update, mu, logvar)

                loss.backward()
                self.optimizer.step()


    def _gen_alpha_space(self, updates):
        noise_vecs = []
        for i in range(len(updates)):
            for j in range(len(updates)):
                if i != j:
                    noise_ij = abs(updates[i].state_dict_oneD - updates[j].state_dict_oneD)
                    noise_vecs.append(noise_ij)
        return noise_vecs


    def regularize_updates(self, updates):
        self.vae.eval()
        self.vae.to('cuda:4')

        for update in updates:
            if update.meta['malf']:
                _update = torch.Tensor(update.state_dict_oneD).to('cuda:4')
                regularized_update, _, _ = self.vae(_update)
                regularized_update       = regularized_update.cpu().numpy()
                update.state_dict_oneD   = regularized_update
                update.oneD_to_state_dict(regularized_update)
        return updates

class UNet1D(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, features=[32, 64, 128]):
        super(UNet1D, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        
        # Downsampling Path (Encoder)
        for feature in features:
            self.encoder.append(self._conv_block(input_channels, feature))
            input_channels = feature
        
        # Bottleneck
        self.bottleneck = self._conv_block(features[-1], features[-1] * 2)
        
        # Upsampling Path (Decoder)
        for feature in reversed(features):
            self.decoder.append(
                nn.ConvTranspose1d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoder.append(self._conv_block(feature * 2, feature))
        
        # Final Convolution
        self.final_conv = nn.Conv1d(features[0], output_channels, kernel_size=1)
    
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        skip_connections = []
        
        # Encoder Path
        for layer in self.encoder:
            x = layer(x)
            skip_connections.append(x)
            x = F.max_pool1d(x, kernel_size=2)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder Path
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)  # Transpose Convolution
            skip_connection = skip_connections[idx // 2]
            
            # If the shapes don't match due to odd dimensions
            if x.shape[-1] != skip_connection.shape[-1]:
                x = F.interpolate(x, size=skip_connection.shape[-1])
            
            # Concatenate along the channel dimension
            x = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx + 1](x)  # Conv Block
        
        # Final Convolution
        return self.final_conv(x)
