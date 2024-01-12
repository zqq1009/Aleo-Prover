use std::{
    sync::{
        Arc,
    },
    time::Duration,
};

use rand::{rngs::OsRng, Rng};
use snarkvm::{
    prelude::{Testnet3, Network},
    ledger::coinbase::EpochChallenge,
};

use tokio::{
    sync::{
        mpsc,
        mpsc::{Receiver, Sender},
        Mutex,
    },
    task,
    time::sleep,
};
use tracing::{debug, error, info, trace, warn};

use crate::ServerMessage;

pub struct Node {
    operator: String,
    // sender: Arc<Sender<SnarkOSMessage>>,
    // receiver: Arc<Mutex<Receiver<SnarkOSMessage>>>,
    sender: Arc<Sender<String>>,
    receiver: Arc<Mutex<Receiver<String>>>,
}

impl Node {
    pub fn init(operator: String) -> Self {
        let (sender, receiver) = mpsc::channel(1024);
        Self {
            operator,
            sender: Arc::new(sender),
            receiver: Arc::new(Mutex::new(receiver)),
        }
    }

    // pub fn receiver(&self) -> Arc<Mutex<Receiver<SnarkOSMessage>>> {
    pub fn receiver(&self) -> Arc<Mutex<Receiver<String>>> {
        self.receiver.clone()
    }

    pub fn sender(&self) -> Arc<Sender<String>> {
        self.sender.clone()
    }
}

pub(crate) type SnarkOSMessage = snarkos_node_router_messages::Message<Testnet3>;


pub fn start(node: Node, server_sender: Sender<ServerMessage>) {
    let receiver = node.receiver();
    task::spawn(async move {

        task::spawn(async move {

            let mut epoch_number = 100;
            let proof_target = 100;
            
            loop {
                epoch_number += 1;

                let rng = &mut OsRng;
                let epoch_block_hash : <Testnet3 as Network>::BlockHash = rng.gen();
                let epoch_challenge = EpochChallenge::<Testnet3>::new(epoch_number, epoch_block_hash, (1 << 10) - 1 ).unwrap();

                if let Err(e) = server_sender.send(ServerMessage::NewEpochChallenge(
                    epoch_challenge, proof_target
                )).await {
                    error!("Error sending new block template to pool server: {}", e);
                } else {
                    trace!("Sent new epoch challenge {} to pool server", epoch_number);
                }

                sleep(Duration::from_secs(120)).await;
            }
        });

        loop {

            let receiver = &mut *receiver.lock().await;
            loop {
                tokio::select! {
                    Some(message) = receiver.recv() => {
                        trace!("dummy Sending {} to validator {}", message, node.operator);
                    }

                }
            }
        }
    });


}
