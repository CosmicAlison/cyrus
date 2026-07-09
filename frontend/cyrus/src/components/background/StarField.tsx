import StarrySky from '@gura_ame/starry-sky';
import '@gura_ame/starry-sky/dist/StarrySky.css';

function StarField() {
  return (
      <StarrySky
        starCount={100}
        meteorInterval={[8000, 13000]}
        showMoon={false}
        showForest={false}
        className="night-sky"
        style={{ zIndex: -1 }}
      />
  );
}

export default StarField;