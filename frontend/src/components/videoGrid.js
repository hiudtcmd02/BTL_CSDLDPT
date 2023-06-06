import React from 'react';
import Modal from 'react-modal';
import { AiOutlineClose } from 'react-icons/ai';
const VideoGridModal = ({ video, onClose }) => {

    const closeModal = () => {
        onClose();
    };
    return (
        <Modal isOpen={true} onRequestClose={onClose}
            style={{
                content: {
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    width: '70%',
                    height: 'auto',
                    marginLeft: 220
                },
            }}
        >
            <button className="close-button" onClick={closeModal}>
                <AiOutlineClose className="close-icon" />
            </button>
            <video src={video.path} type="video/mp4" controls width="90%" height="auto" />
        </Modal>
    );
};
export default VideoGridModal;