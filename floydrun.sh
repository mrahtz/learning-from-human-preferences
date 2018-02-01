git clone https://github.com/openai/gym \
    && cd gym \
    && git reset --hard b5576dc23a5fcad0733042ab2ad440200ebb6209 \
    && pip install -e .[atari] \
    && cd ../gym-gridworld \
    && pip install -e . \
    && cd ../baselines \
    && pip install --upgrade cython cloudpickle && pip install -e . \
    && cd .. \
    && $*
