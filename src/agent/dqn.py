                for _images, _label, _name in tqdm(self.train_loader):
                    # print(index)
                    if next_images is not None:
                        cur_images = next_images
                        cur_label = next_label
                        cur_name = next_name
                        if index == (len(self.train_loader) - 1):
                            next_images, next_label, next_name = next(data_iter)
                        else:
                            next_images = _images
                            next_label = _label
                            next_name = _name
                    else:
                        cur_images = _images
                        cur_label = _label
                        cur_name = _name
                        next_images, next_label, next_name = next(data_iter)
                    images = cur_images.to(self.device)
                    label = cur_label.to(self.device)
                    name = cur_name
                    index += 1
                    cam_extractor = XGradCAM(self.unet, ['module.fire9_1', 'module.fire9_2'])
                    # cam_extractor_e = XGradCAM(self.unet, 'module.e_conv5_x')
                    cam_extractor._hooks_enabled = True
                    # cam_extractor_e._hooks_enabled = True
                    activation_map_g = torch.ones((images.shape[0], 32, 32)).to(self.device)
                    activation_map_e = torch.ones((images.shape[0], 32, 32)).to(self.device)

                    if self.modal == 0:
                        cls = self.unet(images[:, 0:3, :, :])
                    elif self.modal == 1:
                        cls = self.unet(images[:, 3:6, :, :])
                    elif self.modal == 2:
                        cls = self.unet(images[:, 0:6, :, :])
                    elif self.modal == 4:
                        if epoch < self.warmup:
                            cls, g_att_output, e_att_output, x1_1, x2_1, affin_grid_1, affin_grid_2, edge, color = self.unet(
                                torch.zeros((1)), images[:, 0:3, :, :], images[:, 3:6, :, :],
                                activation_map_g,
                                activation_map_e)
                        else:
                            cls, g_att_output, e_att_output, x1_1, x2_1, affin_grid_1, affin_grid_2, edge, color = self.unet(
                                torch.zeros((1)), images[:, 0:3, :, :], images[:, 3:6, :, :],
                                activation_map_g,
                                activation_map_e)
                        # print('*******finish')

                    activation_map = cam_extractor(cls.argmax(axis=1).squeeze(0).tolist(), cls)
                    # activation_map_e = cam_extractor_e(cls.argmax(axis=1).squeeze(0).tolist(), cls)
                    # plt.imshow(activation_map1[0][0].squeeze(0).cpu().numpy())
                    # plt.axis('off')
                    # plt.tight_layout()
                    # plt.show()

                    cam_extractor.remove_hooks()
                    cam_extractor._hooks_enabled = False
                    # cam_extractor_e.remove_hooks()
                    # cam_extractor_e._hooks_enabled = False

                    # print(np.array(activation_map).shape)
                    if self.modal == 4:
                        if epoch < self.warmup:
                            cls, g_att_output, e_att_output, x1_1, x2_1, affin_grid_1, affin_grid_2, edge, color = self.unet(
                                torch.zeros((1)), images[:, 0:3, :, :], images[:, 3:6, :, :],
                                activation_map[0],
                                activation_map[1])
                        else:
                            cls, g_att_output, e_att_output, x1_1, x2_1, affin_grid_1, affin_grid_2, edge, color = self.unet(
                                torch.ones((1)), images[:, 0:3, :, :], images[:, 3:6, :, :],
                                activation_map[0],
                                activation_map[1])
                    cls_cpu = cls.data.cpu().numpy()
                    label_cpu = label.data.cpu().numpy()
                    # print('cls_cpu[:,1]:',cls_cpu[:,1].shape)
                    # print('np.sum(label_cpu==0):',np.array(np.sum(label_cpu==0)).unsqueeze(0).shape)
                    state = np.append(cls_cpu[:, 1],
                                      [np.array(np.sum(label_cpu == 0)), np.array(np.sum(label_cpu == 1))])
                    # print('label_1:',label_cpu)

                    _action1, _action2 = self.dqn.get_action(state)
                    if self.alpha[0, 0] <= 0.1:
                        if _action1 < 2:
                            _action1 = 2
                    if self.alpha[0, 0] >= 0.9:
                        if _action1 > 2:
                            _action1 = 2
                    if self.cls_criterion.gamma <= 0.1:
                        if _action2 < 2:
                            _action2 = 2
                    if self.cls_criterion.gamma >= 3.9:
                        if _action2 > 2:
                            _action2 = 2

                    if epoch > self.warmup:
                        self.alpha[0, 0] += self.action1[_action1]
                        self.alpha[1, 0] -= self.action1[_action1]
                        print('action1:', self.action1[_action1], 'action1:', -self.action1[_action1], 'action2:',
                              self.action1[_action2])

                        self.cls_criterion.alpha = self.alpha
                        self.cls_criterion.gamma = self.cls_criterion.gamma + self.action1[_action2]

                    cls_loss = self.cls_criterion(cls, label.long())
                    loss = cls_loss
                    epoch_loss += loss.item()
                    # Backprop + optimize
                    self.reset_grad()
                    loss.backward()
                    self.optimizer.step()

                    batch_size = label.size(0)

                    total_b = 0
                    true_b = 0
                    tp_b = 0
                    fp_b = 0
                    tn_b = 0
                    fn_b = 0
                    for b in range(batch_size):
                        total_b += 1
                        cls_out = cls_cpu[b].argmax()
                        label_out = label_cpu[b]
                        if cls_out == label_out:
                            true_b += 1
                            if cls_out == 1:
                                tp_b += 1
                            else:
                                tn_b += 1
                        else:
                            if cls_out == 1:
                                fp_b += 1
                            else:
                                fn_b += 1

                    cls_acc_b = true_b / total_b
                    cls_pre_b = tp_b / (tp_b + fp_b + 1e-8)
                    cls_rec_b = tp_b / (tp_b + fn_b + 1e-8)
                    cls_spe_b = tn / (tn_b + fp_b + 1e-8)
                    cls_f1_b = (2 * cls_pre_b * cls_rec_b) / (cls_pre_b + cls_rec_b + 1e-8)
                    images = next_images.to(self.device)
                    label = next_label.to(self.device)
                    # print(images.shape)
                    # print(label.shape)
                    activation_map_g = torch.ones((images.shape[0], 32, 32)).to(self.device)
                    activation_map_e = torch.ones((images.shape[0], 32, 32)).to(self.device)
                    if self.modal == 4:
                        if epoch < self.warmup:
                            cls, g_att_output, e_att_output, x1_1, x2_1, affin_grid_1, affin_grid_2, edge, color = self.unet(
                                torch.zeros((1)), images[:, 0:3, :, :], images[:, 3:6, :, :],
                                activation_map_g,
                                activation_map_e)
                        else:
                            cls, g_att_output, e_att_output, x1_1, x2_1, affin_grid_1, affin_grid_2, edge, color = self.unet(
                                torch.zeros((1)), images[:, 0:3, :, :], images[:, 3:6, :, :],
                                activation_map_g,
                                activation_map_e)
                    # print(cls.shape)
                    activation_map = cam_extractor(cls.argmax(axis=1).squeeze(0).tolist(), cls)
                    # activation_map_e = cam_extractor_e(cls.argmax(axis=1).squeeze(0).tolist(), cls)
                    # plt.imshow(activation_map1[0][0].squeeze(0).cpu().numpy())
                    # plt.axis('off')
                    # plt.tight_layout()
                    # plt.show()

                    cam_extractor.remove_hooks()
                    cam_extractor._hooks_enabled = False

                    if self.modal == 4:
                        if epoch < self.warmup:
                            cls, g_att_output, e_att_output, x1_1, x2_1, affin_grid_1, affin_grid_2, edge, color = self.unet(
                                torch.zeros((1)), images[:, 0:3, :, :], images[:, 3:6, :, :],
                                activation_map[0],
                                activation_map[1])
                        else:
                            cls, g_att_output, e_att_output, x1_1, x2_1, affin_grid_1, affin_grid_2, edge, color = self.unet(
                                torch.ones((1)), images[:, 0:3, :, :], images[:, 3:6, :, :],
                                activation_map[0],
                                activation_map[1])

                    cls_cpu = cls.data.cpu().numpy()
                    label_cpu = label.data.cpu().numpy()
                    # print('cls_cpu[:,1]:',cls_cpu[:,1].shape)
                    # print('np.sum(label_cpu==0):',np.sum(label_cpu==0).shape)
                    # cls_f1_b*cls_f1_b*cls_f1_b+(cls_rec_b+cls_spe_b-1)*(cls_rec_b+cls_spe_b-1)
                    next_state = np.append(cls_cpu[:, 1],
                                           [np.array(np.sum(label_cpu == 0)), np.array(np.sum(label_cpu == 1))])
                    self.dqn.replay_buffer.push(state, [_action1, _action2], (true_b + tn_b), next_state, 1)
                    if len(self.dqn.replay_buffer) > 8:
                        self.dqn.update(8)
                    # cls_out = cls_cpu.argmax()
                    for b in range(batch_size):
                        total += 1

                        name_list.append(name[b])
                        probility_list.append(cls_cpu[b, 1])
                        cls_out = cls_cpu[b].argmax()
                        """if np.exp(cls_cpu[b, 1]) > 0.5:
                            cls_out = 1
                        else:
                            cls_out = 0"""
                        label_out = label_cpu[b]
                        label_list.append(label_out)
                        predict_list.append(cls_out)
                        if cls_out == label_out:
                            true += 1
                            if cls_out == 1:
                                tp += 1
                            else:
                                tn += 1
                        else:
                            if cls_out == 1:
                                fp += 1
                            else:
                                fn += 1

                # self.scheduler.step(epoch_loss)
                self.scheduler.step()
