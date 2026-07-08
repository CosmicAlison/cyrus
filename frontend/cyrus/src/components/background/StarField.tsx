import StarrySky from '@gura_ame/starry-sky';
import '@gura_ame/starry-sky/dist/StarrySky.css';

function StarField() {
  return (
    <div style={{ height: '100vh' }}>
      <StarrySky
        starCount={100}
        meteorInterval={[8000, 13000]}
        showMoon={false}
        showForest={false}
        className="night-sky"
        style={{ zIndex: -1 }}
      />
    </div>
  );
}

export default StarField;