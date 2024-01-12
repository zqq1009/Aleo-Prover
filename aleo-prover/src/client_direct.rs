use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};

use futures_util::sink::SinkExt;
use snarkos_node_executor::{NodeType, Status};
use snarkos_node_messages::{
    ChallengeRequest,
    ChallengeResponse,
    Data,
    MessageCodec,
    Ping,
    Pong,
    PuzzleRequest,
    PuzzleResponse,
};
use snarkvm::{
    console::account::address::Address,
    prelude::{Block, FromBytes, Network, Testnet3},
};
use tokio::{
    net::{TcpListener, TcpStream},
    sync::{
        mpsc,
        mpsc::{Receiver, Sender},
        Mutex,
    },
    task,
    time::{sleep, timeout},
};
use tokio_stream::StreamExt;
use tokio_util::codec::Framed;
use tracing::{debug, error, info, warn};

use crate::prover::ProverEvent;

type Message = snarkos_node_messages::Message<Testnet3>;

pub struct DirectClient {
    pub address: Address<Testnet3>,
    server: String,
    sender: Arc<Sender<Message>>,
    receiver: Arc<Mutex<Receiver<Message>>>,
}

impl DirectClient {
    pub fn init(address: Address<Testnet3>, server: String) -> Arc<Self> {
        let (sender, receiver) = mpsc::channel(1024);
        Arc::new(Self {
            address,
            server,
            sender: Arc::new(sender),
            receiver: Arc::new(Mutex::new(receiver)),
        })
    }

    pub fn sender(&self) -> Arc<Sender<Message>> {
        self.sender.clone()
    }

    pub fn receiver(&self) -> Arc<Mutex<Receiver<Message>>> {
        self.receiver.clone()
    }
}

pub fn start(prover_sender: Arc<Sender<ProverEvent>>, client: Arc<DirectClient>) {
    task::spawn(async move {
        let receiver = client.receiver();
        let genesis_header = *Block::<Testnet3>::from_bytes_le(Testnet3::genesis_bytes())
            .unwrap()
            .header();
        let connected = Arc::new(AtomicBool::new(false));
        let client_sender = client.sender();

        let connected_req = connected.clone();
        task::spawn(async move {
            loop {
                sleep(Duration::from_secs(Testnet3::ANCHOR_TIME as u64)).await;
                if connected_req.load(Ordering::SeqCst) {
                    if let Err(e) = client_sender.send(Message::PuzzleRequest(PuzzleRequest {})).await {
                        error!("Failed to send puzzle request: {}", e);
                    }
                }
            }
        });

        // incoming socket
        task::spawn(async move {
            let (_, listener) = match TcpListener::bind("0.0.0.0:4140").await {
                Ok(listener) => {
                    let local_ip = listener.local_addr().expect("Could not get local ip");
                    info!("Listening on {}", local_ip);
                    (local_ip, listener)
                }
                Err(e) => {
                    panic!("Unable to listen on port 4140: {:?}", e);
                }
            };
            loop {
                match listener.accept().await {
                    Ok((stream, peer_addr)) => {
                        info!("New connection from: {}", peer_addr);
                        // snarkOS is not checking anything so we just hang up
                        drop(stream);
                    }
                    Err(e) => {
                        error!("Error accepting connection: {:?}", e);
                    }
                }
            }
        });

        debug!("Created coinbase puzzle request task");
        loop {
            info!("Connecting to server...");
            match timeout(Duration::from_secs(5), TcpStream::connect(&client.server)).await {
                Ok(socket) => match socket {
                    Ok(socket) => {
                        info!("Connected to {}", client.server);
                        let mut framed = Framed::new(socket, MessageCodec::default());
                        let challenge_request = Message::ChallengeRequest(ChallengeRequest {
                            version: Message::VERSION,
                            fork_depth: 4096,
                            node_type: NodeType::Prover,
                            status: Status::Ready,
                            listener_port: 4140,
                        });
                        if let Err(e) = framed.send(challenge_request).await {
                            error!("Error sending challenge request: {}", e);
                        } else {
                            debug!("Sent challenge request");
                        }
                        let receiver = &mut *receiver.lock().await;
                        loop {
                            tokio::select! {
                                Some(message) = receiver.recv() => {
                                    let m = message.clone();
                                    let name = m.name();
                                    debug!("Sending {} to beacon", name);
                                    if let Err(e) = framed.send(message).await {
                                        error!("Error sending {}: {:?}", name, e);
                                    }
                                }
                                result = framed.next() => match result {
                                    Some(Ok(message)) => {
                                        debug!("Received {} from beacon", message.name());
                                        match message {
                                            Message::ChallengeRequest(ChallengeRequest {
                                                version,
                                                fork_depth: _,
                                                node_type,
                                                ..
                                            }) => {
                                                if version < Message::VERSION {
                                                    error!("Peer is running an older version of the protocol");
                                                    sleep(Duration::from_secs(5)).await;
                                                    break;
                                                }
                                                if node_type != NodeType::Beacon && node_type != NodeType::Validator {
                                                    error!("Peer is not a beacon or validator");
                                                    sleep(Duration::from_secs(5)).await;
                                                    break;
                                                }
                                                let response = Message::ChallengeResponse(ChallengeResponse {
                                                    header: Data::Object(genesis_header),
                                                });
                                                if let Err(e) = framed.send(response).await {
                                                    error!("Error sending challenge response: {:?}", e);
                                                } else {
                                                    debug!("Sent challenge response");
                                                }
                                            }
                                            Message::ChallengeResponse(message) => {
                                                // Perform the deferred non-blocking deserialization of the block header.
                                                let block_header = match message.header.deserialize().await {
                                                    Ok(block_header) => block_header,
                                                    Err(error) => {
                                                        error!("Error deserializing block header: {:?}", error);
                                                        sleep(Duration::from_secs(5)).await;
                                                        break;
                                                    }
                                                };
                                                match block_header == genesis_header {
                                                    true => {
                                                        // Send the first `Ping` message to the peer.
                                                        let message = Message::Ping(Ping {
                                                            version: Message::VERSION,
                                                            fork_depth: 4096,
                                                            node_type: NodeType::Prover,
                                                            status: Status::Ready,
                                                        });
                                                        if let Err(e) = framed.send(message).await {
                                                            error!("Error sending ping: {:?}", e);
                                                        } else {
                                                            debug!("Sent ping");
                                                        }
                                                    }
                                                    false => {
                                                        error!("Peer has a different genesis block");
                                                        sleep(Duration::from_secs(5)).await;
                                                        break;
                                                    }
                                                }
                                            }
                                            Message::Ping(_) => {
                                                let pong = Message::Pong(Pong { is_fork: None });
                                                if let Err(e) = framed.send(pong).await {
                                                    error!("Error sending pong: {:?}", e);
                                                } else {
                                                    debug!("Sent pong");
                                                }
                                            }
                                            Message::Pong(_) => {
                                                connected.store(true, Ordering::SeqCst);
                                                if let Err(e) = client.sender().send(Message::PuzzleRequest(PuzzleRequest {})).await {
                                                    error!("Failed to send puzzle request: {}", e);
                                                }
                                            }
                                            Message::PuzzleResponse(PuzzleResponse {
                                                epoch_challenge, block
                                            }) => {
                                                let block = match block.deserialize().await {
                                                    Ok(block) => block,
                                                    Err(error) => {
                                                        error!("Error deserializing block: {:?}", error);
                                                        sleep(Duration::from_secs(5)).await;
                                                        break;
                                                    }
                                                };
                                                if let Err(e) = prover_sender.send(ProverEvent::NewTarget(block.proof_target())).await {
                                                    error!("Error sending new target to prover: {}", e);
                                                } else {
                                                    debug!("Sent new target to prover");
                                                }
                                                if let Err(e) = prover_sender.send(ProverEvent::NewWork(epoch_challenge.epoch_number(), epoch_challenge, client.address)).await {
                                                    error!("Error sending new work to prover: {}", e);
                                                } else {
                                                    debug!("Sent new work to prover");
                                                }
                                            }
                                            Message::Disconnect(message) => {
                                                error!("Peer disconnected: {:?}", message.reason);
                                                sleep(Duration::from_secs(5)).await;
                                                break;
                                            }
                                            _ => {
                                                debug!("Unhandled message: {}", message.name());
                                            }
                                        }
                                    }
                                    Some(Err(e)) => {
                                        warn!("Failed to read the message: {:?}", e);
                                    }
                                    None => {
                                        error!("Disconnected from beacon");
                                        connected.store(false, Ordering::SeqCst);
                                        sleep(Duration::from_secs(5)).await;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => {
                        error!("Failed to connect to beacon: {}", e);
                        sleep(Duration::from_secs(5)).await;
                    }
                },
                Err(_) => {
                    error!("Failed to connect to beacon: Timed out");
                    sleep(Duration::from_secs(5)).await;
                }
            }
        }
    });
}
