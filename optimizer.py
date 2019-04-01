class SGVB_Optimizer(Base_Optimizer):
    
    def __init__(self, model, loss, data_iterator, session, l_rate=1e-4, save_path=None, load_path=None):
        """
        Implements the Stochastic Gradient Variational Bayes optimizer for bayesian training.
        
        Args:
            model: bayesian.model, the model to perform bayesian inference on
            data_iterator: the dataset iterator used for training
            session: tf.session, the session used to run the optimization in
            l_rate: float, the learning rate for the sgd updates
        
        Returns:
            class instance of SGVB_Optimizer
        """        
        self.model          = model
        self.session        = session
        self.data_iterator  = data_iterator
        self.l_rate         = l_rate
        self.save_path      = save_path   
        self.loss           = loss
            
        self.data_sample    = self.data_iterator.get_next()
        self.initialize()
        
        #set up the gradient descent optimizer
        print("setting up the optimizer")
        self.optimizer      = tf.train.GradientDescentOptimizer(self.l_rate)
        self.model_params   = tf.trainable_variables()
        self.train          = self.optimizer.minimize(self.loss, var_list=self.model_params)
        
        self.load_path = load_path
        
        #load from checkpoint if loadpath is given
        if load_path is not None:
            self.model.load(self.session, self.load_path + "_model.ckpt") #TODO: implement model.load
        
        #create the summary
        self.summary        = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(self.save_path, self.session.graph)  
    
    def initialize(self):
        """
        Initializes all the variables and iterators needed for optimization
        """
        
        #initialize the variables
        init = tf.global_variables_initializer()
        self.session.run(init)
        
        #initialize the data iterators
        self.session.run(self.data_iterator.initializer)
        
    def optimize(self, n_epochs, verbose=True, save=False, n_save=20, evaluation=None, test_data=None, test_iterator_init=None):
            """
            Applies the SGVB updates.
            
            Args:
                verbose: boolean, print the training process
                n_epochs: integer, the number of epochs
            
            Returns:
                current loss after update
            """
            loss_vals   = [0]
            acc         = [0]
            counter     = 0
            #summary     = self.session.run(self.summary)
            
            for ep in range(n_epochs):
                #re-initialize the iterator on the dataset
                self.session.run(self.data_iterator.initializer)
                while True:
                    try:
                        #perfrom the update-step
                        print("batch update")
                        _, loss_value = self.session.run((self.train, self.loss))
                        print(loss_value)
                        loss_vals.append(loss_value)
                        
                    except tf.errors.OutOfRangeError:
                        break
                    
                #self.summary_writer.add_summary(summary, counter)
                counter += 1
                    
                ##sample new realizations
                #self.session.run(self.samplers)
                #ns = self.session.run(self.model.param_realizations)
                ##write them one by one into the 
                #for si, nsi in zip(samples, ns):
                    #si.append(nsi)
                
                ##calculate the accuracy
                #if evaluation is not None:
                    #self.session.run(test_iterator_init)
                    #l_mean      = evaluation(test_data, [si[-1:] for si in samples])
                    #res = []
                    #ref = []
                    #while True:
                        #try:
                            #r,y = self.session.run((l_mean, test_data['y']))
                            #res.append(r)
                            #ref.append(y)
                        #except tf.errors.OutOfRangeError:
                            #break
                        
                    #res=np.argmax(np.concatenate(res, axis=0),axis = -1)
                    #ref=np.argmax(np.concatenate(ref, axis=0),axis = -1)
                    #acc.append(np.float(np.sum(res==ref)) / len(ref))
                    
                #if verbose:
                    #if evaluation is not None:
                        #print("SVGD in epoch: " + str(ep) + " with cost: "+str(loss_value) +" with accuracy: " + str(acc[-1]))
                    #else:
                        #print("SVGD in epoch: " + str(ep) + " with cost: "+str(loss_value))
                    
                #if save == True:
                    ##save the realizations
                    #with open(self.save_path+"_samples.pkl", "wb") as fp:  
                        #pickle.dump(samples[-n_save:], fp)
                    #fp.close()
                    ##save the loss    
                    #with open(self.save_path+"_loss.pkl", "wb") as fp:  
                        #pickle.dump(loss_vals, fp)
                    #fp.close()
                    #if evaluation is not None:
                        ##save the accuracy
                        #with open(self.save_path+"_acc.pkl", "wb") as fp:  
                            #pickle.dump(acc, fp)
                        #fp.close()
            return loss_vals, samples






